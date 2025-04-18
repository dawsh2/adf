# Show current branch
echo "Current branch: $(git branch --show-current)"

# Show git status first so user can see what will be committed
echo -e "\nFiles to be committed:"
git status -s

# Prompt for commit message
echo -e "\nEnter commit message:"
read commit_message

# Check if commit message is empty
if [ -z "$commit_message" ]; then
  echo "Commit message cannot be empty. Using default message."
  commit_message="Update files in $(git branch --show-current) branch"
fi

# Stage all changes
echo -e "\nStaging all changes..."
git add .

# Commit with the provided message
echo "Committing changes with message: \"$commit_message\""
git commit -m "$commit_message"

# Push to the current branch
current_branch=$(git branch --show-current)
echo "Pushing to origin/$current_branch..."
git push origin "$current_branch"

echo -e "\nDone!"
