# GitHub Issue Analysis Tools

Python tools for downloading and analyzing GitHub issues, with special focus on ROS (Robot Operating System) related problems.

## Tools
### Issue Parser
Downloads complete GitHub issue threads with attachments and conversation history.

### Issue Analyzer
Processes downloaded issues to identify terminal nodes and failure points in ROS systems.

## Requirements
- Python 3.8+
- Required packages:
  ```
  requests
  openai
  tenacity
  python-dotenv
  ```

## Usage

### 1. Parse GitHub Issues
Create an input JSON file with issue URLs:
```json
[
    "https://github.com/owner/repo/issues/123",
    "https://github.com/owner/repo/issues/456"
]
```

Run the parser:
```bash
python parser.py -i input.json -o output -v
```

### 2. Analyze Issues
After parsing, analyze the issues:
```bash
python analyzer.py -i output -v
```

## Arguments

### Parser Arguments
- `-i, --input`: Input JSON file with issue URLs
- `-o, --output`: Output directory path
- `-t, --token`: GitHub token (optional)
- `-v, --verbose`: Enable verbose output

### Analyzer Arguments
- `-i, --input`: Directory containing parsed issues
- `-k, --api-key`: OpenAI API key (or use OPENAI_API_KEY env var)
- `-p, --prompt`: Path to prompt template (default: prompt.txt)
- `-m, --model`: Model to use (default: gpt-4o)
- `-v, --verbose`: Enable verbose output

## Output Structure
```
output/
└── issue_123/
    ├── issue-full.md        # Consolidated issue thread
    ├── issue.md            # Original issue
    ├── metadata.json       # Issue metadata
    ├── comments/           # Individual comments
    ├── media/             # Downloaded assets
    └── analysis/          # Analysis results
        └── terminal_nodes_analysis_20240304_123456.json
```

## Use Cases
- Archiving GitHub issues
- ROS issue triage
- System dependency analysis
- Failure point identification