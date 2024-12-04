import os
import json
import base64
import asyncio
import argparse
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when there's an issue with configuration."""
    pass


class AnalysisError(Exception):
    """Raised when there's an issue with the analysis process."""
    pass


class ROSIssueAnalyzer:
    def __init__(
        self,
        api_key: str,
        prompt_template: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        image_detail: str = "high",
        verbose: bool = False,
    ):
        """Initialize the ROS Issue Analyzer."""
        self.client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_detail = image_detail
        self.logger = self._setup_logger(verbose)
        self.executor = ThreadPoolExecutor(max_workers=4)

    @classmethod
    async def create(
        cls, api_key: str, prompt_path: Path, **kwargs
    ) -> "ROSIssueAnalyzer":
        """Factory method to create analyzer instance."""
        try:
            if not prompt_path.exists():
                raise ConfigurationError(f"Prompt template file not found: {prompt_path}")
            
            prompt_template = await cls._read_file_async(prompt_path)
            return cls(api_key=api_key, prompt_template=prompt_template, **kwargs)
        
        except Exception as e:
            logging.error(f"Failed to create analyzer: {e}")
            raise

    @staticmethod
    async def _read_file_async(file_path: Path) -> str:
        """Asynchronously read file content."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: file_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to read file {file_path}: {e}")

    def _setup_logger(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("ROSIssueAnalyzer")
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.handlers.clear()

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ros_analyzer_{timestamp}.log"

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    async def _encode_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Encode image to base64 with validation."""
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                self.executor, image_path.stat
            )

            if stats.st_size > 20 * 1024 * 1024:  # 20MB limit
                self.logger.warning(f"Image {image_path} exceeds 20MB limit, skipping")
                return None

            valid_formats = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            if image_path.suffix.lower() not in valid_formats:
                self.logger.warning(f"Unsupported image format for {image_path}, skipping")
                return None

            base64_image = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: base64.b64encode(image_path.read_bytes()).decode("utf-8"),
            )

            mime_type = f"image/{image_path.suffix[1:].lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": self.image_detail,
                },
            }

        except Exception as e:
            self.logger.warning(f"Failed to encode image {image_path}: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        before_sleep=before_sleep_log(logging.getLogger("ROSIssueAnalyzer"), logging.WARNING)
    )
    async def _analyze_issue(
        self, content: str, images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send issue to OpenAI API for analysis."""
        try:
            message_content = [
                {
                    "type": "text",
                    "text": self.prompt_template.format(issue_content=content),
                }
            ]
            message_content.extend(images)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a ROS expert analyzing issue threads to identify terminal nodes. "
                        "You must respond with a valid JSON object only, following the exact format "
                        "specified in the prompt. Start your response with '{' and end with '}'. "
                        "Do not include any additional text or formatting."
                    )
                },
                {"role": "user", "content": message_content},
            ]

            self.logger.debug("Sending request to OpenAI API...")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            raw_content = response.choices[0].message.content
            self.logger.debug(f"Raw API response: {raw_content}")

            # Clean and extract JSON from response
            cleaned_content = raw_content.strip()
            start_idx = cleaned_content.find('{')
            end_idx = cleaned_content.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise AnalysisError("No valid JSON object found in response")
            
            json_str = cleaned_content[start_idx:end_idx + 1]
            
            try:
                analysis = json.loads(json_str)
                self.logger.debug(f"Parsed JSON: {json.dumps(analysis, indent=2)}")
                
                # Validate required fields
                required_fields = {
                    "analysis_timestamp": str,
                    "terminal_nodes": list,
                    "analysis_confidence": dict,
                    "verification_steps": list
                }
                
                for field, expected_type in required_fields.items():
                    if field not in analysis:
                        raise AnalysisError(f"Missing required field: {field}")
                    if not isinstance(analysis[field], expected_type):
                        raise AnalysisError(
                            f"Invalid type for {field}: expected {expected_type.__name__}, "
                            f"got {type(analysis[field]).__name__}"
                        )

                # Validate terminal_nodes structure if present
                if analysis["terminal_nodes"]:
                    for i, node in enumerate(analysis["terminal_nodes"]):
                        if not isinstance(node, dict):
                            raise AnalysisError(
                                f"Invalid terminal node at index {i}: expected dict, "
                                f"got {type(node).__name__}"
                            )
                        
                        required_node_fields = {
                            "node_name": str,
                            "terminal_classification": str,
                            "terminal_evidence": list,
                            "component_type": str,
                            "interfaces": dict,
                            "upstream_dependencies": list
                        }
                        
                        for field, expected_type in required_node_fields.items():
                            if field not in node:
                                raise AnalysisError(f"Missing field '{field}' in terminal node {i}")
                            if not isinstance(node[field], expected_type):
                                raise AnalysisError(
                                    f"Invalid type for {field} in terminal node {i}: "
                                    f"expected {expected_type.__name__}, "
                                    f"got {type(node[field]).__name__}"
                                )

                return analysis

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {e}")
                self.logger.error(f"Attempted to parse: {json_str}")
                raise AnalysisError(f"Failed to parse API response as JSON: {str(e)}")

        except asyncio.TimeoutError:
            self.logger.error("API request timed out")
            raise
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)
            raise

    async def analyze_issue_directory(
        self, issue_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single issue directory."""
        try:
            issue_file = issue_dir / "issue-full.md"
            if not issue_file.exists():
                raise FileNotFoundError(f"issue-full.md not found in {issue_dir}")

            self.logger.info(f"Processing issue directory: {issue_dir}")
            content = await self._read_file_async(issue_file)

            media_dir = issue_dir / "media"
            images = []
            if media_dir.exists():
                image_tasks = []
                for img_path in sorted(media_dir.glob("*")):
                    image_tasks.append(self._encode_image(img_path))

                image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
                images = [result for result in image_results if isinstance(result, dict)]

            self.logger.info(f"Analyzing issue with {len(images)} images")
            analysis = await self._analyze_issue(content, images)

            analysis_dir = issue_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            analysis_file = analysis_dir / f"terminal_nodes_analysis_{timestamp}.json"

            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: analysis_file.write_text(
                    json.dumps(analysis, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                ),
            )

            self.logger.info(f"Analysis saved to {analysis_file}")
            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze issue directory {issue_dir}: {e}")
            raise


async def get_api_key(args_key: Optional[str]) -> str:
    """Get API key from args or environment."""
    if args_key:
        return args_key

    load_dotenv()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    raise ConfigurationError(
        "OpenAI API key not found. Please provide it via -k argument "
        "or set OPENAI_API_KEY environment variable."
    )


async def main():
    """Main function to run the analyzer."""
    parser = argparse.ArgumentParser(description="ROS Issue LLM Analyzer")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory containing issue directories",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        help="OpenAI API key (optional if OPENAI_API_KEY env var is set)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=Path,
        default=Path("prompt.txt"),
        help="Path to prompt template file",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for API calls"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for response"
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="high",
        help="Detail level for image processing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    args = parser.parse_args()

    try:
        api_key = await get_api_key(args.api_key)

        analyzer = await ROSIssueAnalyzer.create(
            api_key=api_key,
            prompt_path=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            image_detail=args.image_detail,
            verbose=args.verbose,
        )

        input_dir = Path(args.input)
        if not input_dir.exists():
            raise ConfigurationError(f"Input directory not found: {input_dir}")

        issues = sorted(input_dir.glob("issue_*"))
        if not issues:
            raise ConfigurationError(f"No issue directories found in {input_dir}")

        tasks = [
            analyzer.analyze_issue_directory(issue_dir)
            for issue_dir in issues
            if issue_dir.is_dir()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, dict))
        error_count = sum(1 for r in results if isinstance(r, Exception))

        print("\nAnalysis Summary:")
        print(f"Total issues processed: {len(tasks)}")
        print(f"Successful analyses: {success_count}")
        print(f"Failed analyses: {error_count}")

        if error_count > 0:
            print("\nErrors encountered:")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Issue #{i+1}: {str(result)}")

        return 1 if error_count > 0 else 0

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        print(f"Script execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Configure asyncio to use better selector on Windows
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run main program
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        exit(130)