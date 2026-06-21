# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly
3. Provide detailed information about the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

- FlashVLM loads model weights from disk or HuggingFace Hub. Only load models from trusted sources.
- When using the CLI or API server, ensure proper access controls are in place.
- Model outputs should not be treated as ground truth for safety-critical applications.

## Best Practices

- Keep dependencies updated to their latest secure versions
- Use virtual environments to isolate dependencies
- Review model sources before loading untrusted weights
- Implement rate limiting when deploying as a service
