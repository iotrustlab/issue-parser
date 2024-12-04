# GitHub Issue Parser

A Python tool for downloading GitHub issues with their complete conversation threads and media attachments.

## Features
- Downloads complete issue threads including comments
- Preserves all images and attachments
- Creates consolidated markdown files for each issue
- Handles GitHub asset URL redirects
- Maintains relative paths for media files

## Requirements
- Python 3.8+
- `requests` library (install using `pip install requests`)

## Usage

1. Create an input JSON file with GitHub issue URLs:
```json
[
    "https://github.com/owner/repo/issues/123",
    "https://github.com/owner/repo/issues/456"
]
```

2. Run the script:
```bash
python main.py -i input.json -o output -v
```

Arguments:
- `-i, --input`: Input JSON file path
- `-o, --output`: Output directory path
- `-t, --token`: GitHub token (optional)
- `-v, --verbose`: Enable verbose output

## Output Structure
```bash
output/
└── issue_123/
    ├── issue-full.md      # Consolidated issue thread
    ├── issue.md           # Original issue
    ├── metadata.json      # Issue metadata
    ├── comments/          # Individual comments
    └── media/            # Downloaded assets
```

## Use Cases
- Archiving GitHub issues
- Creating issue datasets for research
