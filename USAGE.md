# Command Line Usage

## classify_room_type.py - Command Line Arguments

### Usage

```bash
# Run for all projects (default)
uv run python classifier/classify_room_type.py

# Run in test mode (only Project 10)
uv run python classifier/classify_room_type.py --test

# Show help
uv run python classifier/classify_room_type.py --help
```

### Arguments

- `--test` - Run in test mode, processing only Project 10
  - Useful for quick testing without processing all projects
  - Faster iteration during development
