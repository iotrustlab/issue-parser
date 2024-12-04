import os
import re
import json
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse, unquote
from datetime import datetime


class GitHubAssetDownloader:
    def __init__(self, github_token: Optional[str] = None, verbose: bool = False):
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Asset-Downloader/1.0",
            "Referer": "https://github.com"
        }
        if github_token:
            self.headers["Authorization"] = f"Bearer {github_token}"
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _extract_assets(self, content: str) -> List[str]:
        """Extract GitHub user asset URLs."""
        patterns = [
            r'https://github\.com/user-(?:images|attachments)/[^\s\)\"\']+\b',
            r'https://user-images\.githubusercontent\.com/[^\s\)\"\']+\b'
        ]
        assets = []
        for pattern in patterns:
            assets.extend(re.findall(pattern, content))
        return list(set(assets))  # Remove duplicates

    def _safe_filename(self, url: str, prefix: str = "") -> str:
        """Generate a safe filename from URL preserving extension."""
        parsed = urlparse(url)
        path_parts = parsed.path.split('/')
        
        # Try to find asset ID and extension
        asset_id = None
        for part in path_parts:
            if len(part) >= 32 and '-' in part:
                asset_id = part
                break
        
        # If we found an asset ID, use it as the base name
        if asset_id:
            base_name = asset_id
        else:
            base_name = path_parts[-1] if path_parts else 'asset'
        
        # Clean the filename
        base_name = re.sub(r'[^\w\-_.]', '_', unquote(base_name))
        
        # Add prefix if provided
        if prefix:
            base_name = f"{prefix}_{base_name}"

        # Add .png extension if no extension present
        if '.' not in base_name:
            base_name += '.png'
            
        return base_name

    def download_asset(self, url: str, output_path: str) -> bool:
        """Download a GitHub asset with proper context preservation."""
        try:
            if self.verbose:
                print(f"[INFO] Processing asset URL: {url}")

            # First request to get the redirect
            response = self.session.get(
                url, 
                allow_redirects=False,
                headers={"Accept": "image/*, */*"}
            )
            
            final_url = url
            if response.status_code in (301, 302):
                final_url = response.headers['Location']
                if self.verbose:
                    print(f"[INFO] Following redirect to: {final_url}")
            
            # Download the actual asset
            response = self.session.get(final_url, stream=True)
            response.raise_for_status()

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the file with progress indication
            file_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f:
                if file_size:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if self.verbose:
                                progress = (downloaded / file_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                    if self.verbose:
                        print()  # New line after progress
                else:
                    # If no content length header, just download
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            if self.verbose:
                print(f"[SUCCESS] Downloaded to: {output_path}")
            return True

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to download {url}: {e}")
            return False


def format_datetime(dt_str: str) -> str:
    """Format datetime string to a more readable format."""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%B %d, %Y at %H:%M UTC")
    except:
        return dt_str


def create_consolidated_issue(
    issue_data: dict,
    comments: list,
    media_dir: Path,
    output_path: Path,
    downloader: GitHubAssetDownloader
) -> None:
    """Create a consolidated markdown file containing the full issue thread."""
    try:
        # Start with issue metadata and labels
        content = [
            f"# {issue_data['title']}",
            "",
            "## Issue Metadata",
            f"- **Issue Number:** #{issue_data['number']}",
            f"- **Created By:** @{issue_data['user']['login']}",
            f"- **Created At:** {format_datetime(issue_data['created_at'])}",
            f"- **Last Updated:** {format_datetime(issue_data['updated_at'])}",
            f"- **State:** {issue_data['state']}"
        ]

        # Add labels if present
        if issue_data.get('labels'):
            content.append("- **Labels:** " + ", ".join(label['name'] for label in issue_data['labels']))

        # Add issue description
        content.extend([
            "",
            "## Description",
            "",
            issue_data['body'] or "*No description provided*",
            ""
        ])

        # Add issue attachments if any
        issue_assets = downloader._extract_assets(issue_data['body'] or "")
        if issue_assets:
            content.extend([
                "### Issue Attachments",
                ""
            ])
            for url in issue_assets:
                filename = downloader._safe_filename(url)
                content.append(f"![{filename}](media/{filename})")
                content.append(f"[{filename}](media/{filename})")
                content.append("")

        # Add comments in chronological order
        if comments:
            content.extend([
                "## Discussion Thread",
                ""
            ])
            
            for comment in comments:
                # Add comment metadata and body
                content.extend([
                    f"### Comment by @{comment['user']['login']}",
                    f"*Posted on {format_datetime(comment['created_at'])}*",
                    "",
                    comment['body'] or "*No content*",
                    ""
                ])

                # Add comment attachments if any
                comment_assets = downloader._extract_assets(comment['body'] or "")
                if comment_assets:
                    content.append("#### Attachments")
                    content.append("")
                    for url in comment_assets:
                        filename = downloader._safe_filename(url)
                        content.append(f"![{filename}](media/{filename})")
                        content.append(f"[{filename}](media/{filename})")
                        content.append("")

                content.append("---")
                content.append("")

        # Write the consolidated content
        output_path.write_text("\n".join(content), encoding='utf-8')

    except Exception as e:
        print(f"[ERROR] Failed to create consolidated issue file: {e}")


def process_issue(url: str, output_dir: str, github_token: Optional[str] = None, verbose: bool = False):
    """Process a GitHub issue and its assets."""
    try:
        # Extract issue details from URL
        parts = url.rstrip("/").split("/")
        owner, repo, issue_number = parts[-4], parts[-3], parts[-1]
        
        # Setup API URLs
        api_base = "https://api.github.com"
        issue_api_url = f"{api_base}/repos/{owner}/{repo}/issues/{issue_number}"
        
        # Initialize downloader
        downloader = GitHubAssetDownloader(github_token, verbose)
        
        if verbose:
            print(f"\n[INFO] Processing Issue #{issue_number}")
        
        # Fetch issue data
        response = downloader.session.get(issue_api_url)
        response.raise_for_status()
        issue_data = response.json()
        
        # Create issue directory structure
        issue_dir = Path(output_dir) / f"issue_{issue_number}"
        media_dir = issue_dir / "media"
        comments_dir = issue_dir / "comments"
        os.makedirs(media_dir, exist_ok=True)
        os.makedirs(comments_dir, exist_ok=True)
        
        # Process issue body and assets
        if issue_data.get("body"):
            # Save separate issue file
            issue_path = issue_dir / "issue.md"
            issue_path.write_text(
                f"# {issue_data['title']}\n\n{issue_data['body']}", 
                encoding='utf-8'
            )
            
            # Download issue assets
            assets = downloader._extract_assets(issue_data["body"])
            for i, asset_url in enumerate(assets, 1):
                filename = downloader._safe_filename(asset_url, f"issue_{issue_number}_{i}")
                output_path = media_dir / filename
                downloader.download_asset(asset_url, str(output_path))
        
        # Fetch and process comments
        comments = []
        if issue_data.get("comments_url"):
            response = downloader.session.get(issue_data["comments_url"])
            response.raise_for_status()
            comments = response.json()
            
            for i, comment in enumerate(comments, 1):
                # Save separate comment file
                comment_path = comments_dir / f"comment_{i}.md"
                comment_path.write_text(
                    f"### {comment['user']['login']} commented on {comment['created_at']}:\n\n{comment['body']}", 
                    encoding='utf-8'
                )
                
                # Download comment assets
                if comment.get("body"):
                    assets = downloader._extract_assets(comment["body"])
                    for j, asset_url in enumerate(assets, 1):
                        filename = downloader._safe_filename(asset_url, f"comment_{i}_{j}")
                        output_path = media_dir / filename
                        downloader.download_asset(asset_url, str(output_path))

        # Create consolidated file
        consolidated_path = issue_dir / "issue-full.md"
        if verbose:
            print(f"[INFO] Creating consolidated issue file: {consolidated_path}")
        
        create_consolidated_issue(
            issue_data=issue_data,
            comments=comments,
            media_dir=media_dir,
            output_path=consolidated_path,
            downloader=downloader
        )
        
        # Save metadata
        metadata = {
            "number": issue_number,
            "title": issue_data["title"],
            "author": issue_data["user"]["login"],
            "created_at": issue_data["created_at"],
            "updated_at": issue_data["updated_at"],
            "state": issue_data["state"],
            "comments_count": issue_data["comments"],
            "url": url
        }
        
        metadata_path = issue_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        if verbose:
            print(f"[INFO] Completed processing Issue #{issue_number}")
        
    except requests.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to process issue: {e}")


def main():
    parser = argparse.ArgumentParser(description="GitHub Issue Parser and Asset Downloader")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file with issue URLs")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-t", "--token", help="GitHub token (optional)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    try:
        # Load input URLs
        with open(args.input, 'r') as f:
            issue_urls = json.load(f)
        
        # Process each issue
        for url in issue_urls:
            process_issue(url, args.output, args.token, args.verbose)
        
        print("\n[INFO] All issues processed")
        
    except Exception as e:
        print(f"[ERROR] Script execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())