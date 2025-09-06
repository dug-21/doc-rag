# Archmemory Example Usage

## Real-world example: React project

```bash
# After installation, mark your key directories
touch src/components/.archtrack
touch src/hooks/.archtrack
touch src/api/.archtrack
touch src/utils/.archtrack
```

## What happens next?

### When you edit `src/components/Button.jsx`:
Claude automatically understands:
- This is the UI components directory
- It follows React component patterns
- It likely exports reusable UI elements

### When you edit `src/api/users.js`:
Claude remembers:
- This handles API endpoints
- It manages user-related business logic
- It probably integrates with your backend

### When you start a new session tomorrow:
Claude already knows:
- Your project structure
- What each directory is responsible for
- The patterns you use in each area

## No configuration needed!

Unlike the complex original version, you don't need to:
- Write JSON configuration files
- Define interfaces or contracts
- Manually document dependencies
- Run multiple analysis scripts

Just `touch .archtrack` and let Claude learn as you work.

## Tips

1. **Start small**: Mark 3-4 directories initially
2. **Think architectural**: Mark directories that define your app's structure
3. **Be selective**: Not every directory needs tracking
4. **Trust the learning**: Claude figures out patterns from your actual code

## Common patterns to mark

```bash
# Frontend projects
touch src/components/.archtrack   # UI components
touch src/pages/.archtrack        # Page components
touch src/hooks/.archtrack        # Custom React hooks
touch src/store/.archtrack        # State management

# Backend projects
touch src/routes/.archtrack       # API routes
touch src/models/.archtrack       # Data models
touch src/services/.archtrack     # Business logic
touch src/middleware/.archtrack   # Express middleware

# Full-stack projects
touch client/src/.archtrack       # Frontend root
touch server/src/.archtrack       # Backend root
touch shared/.archtrack           # Shared code
```