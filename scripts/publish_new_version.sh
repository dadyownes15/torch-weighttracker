#!/usr/bin/env bash
set -euo pipefail

# Release automation must be non-interactive. Keep Git from opening an editor if
# a command path changes or a future edit drops an explicit message flag.
export GIT_EDITOR=true
export GIT_SEQUENCE_EDITOR=true

usage() {
  cat <<'USAGE'
Usage:
  scripts/publish_new_version.sh [options] "commit message"

Commits the current working tree, creates a new version tag, and pushes both the
current branch and the tag. Versions are provided by setuptools-scm from Git tags.
If CI publishes on version tags, the tag push triggers package publication.
If the tag already exists locally on HEAD, the script resumes by pushing it.

Options:
  --bump patch|minor|major  Version part to increment. Default: patch.
  --version X.Y.Z           Use an explicit version instead of bumping.
  --remote NAME             Git remote to push to. Default: origin.
  --skip-tests              Do not run pytest before committing and tagging.
  --skip-fetch              Do not fetch remote tags before choosing a version.
  --dry-run                 Print commands without executing them.
  -h, --help                Show this help.

Environment:
  TAG_PREFIX                Tag prefix to create and inspect. Default: v.

Examples:
  scripts/publish_new_version.sh "Improve README"
  scripts/publish_new_version.sh --bump minor "Add QAT metrics"
  scripts/publish_new_version.sh --version 0.2.0 "Release 0.2.0"
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_cmd "$@"
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

validate_version() {
  [[ "$1" =~ ^[0-9]+[.][0-9]+[.][0-9]+$ ]]
}

bump_version() {
  local version="$1"
  local bump="$2"
  local major minor patch

  IFS=. read -r major minor patch <<< "$version"

  case "$bump" in
    major)
      major=$((major + 1))
      minor=0
      patch=0
      ;;
    minor)
      minor=$((minor + 1))
      patch=0
      ;;
    patch)
      patch=$((patch + 1))
      ;;
    *)
      die "unsupported bump type: $bump"
      ;;
  esac

  printf '%d.%d.%d\n' "$major" "$minor" "$patch"
}

latest_local_version() {
  local prefix="$1"
  local tag version

  while IFS= read -r tag; do
    version="${tag#"$prefix"}"
    if validate_version "$version"; then
      printf '%s\n' "$version"
      return 0
    fi
  done < <(git tag --list "${prefix}[0-9]*.[0-9]*.[0-9]*" --sort=-v:refname)

  printf '0.0.0\n'
}

remote="origin"
bump="patch"
explicit_version=""
run_tests=1
fetch_tags=1
dry_run=0
tag_prefix="${TAG_PREFIX:-v}"
message_parts=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bump)
      [[ $# -ge 2 ]] || die "--bump requires patch, minor, or major"
      bump="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 ]] || die "--version requires X.Y.Z"
      explicit_version="$2"
      shift 2
      ;;
    --remote)
      [[ $# -ge 2 ]] || die "--remote requires a remote name"
      remote="$2"
      shift 2
      ;;
    --skip-tests)
      run_tests=0
      shift
      ;;
    --skip-fetch)
      fetch_tags=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        message_parts+=("$1")
        shift
      done
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      message_parts+=("$1")
      shift
      ;;
  esac
done

[[ ${#message_parts[@]} -gt 0 ]] || {
  usage
  exit 2
}

case "$bump" in
  major|minor|patch) ;;
  *) die "unsupported bump type: $bump" ;;
esac

if [[ -n "$explicit_version" ]] && ! validate_version "$explicit_version"; then
  die "--version must use X.Y.Z, got: $explicit_version"
fi

commit_message="${message_parts[*]}"

git_root="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die "must be run inside a Git repository"
cd "$git_root"

branch="$(git branch --show-current)"
[[ -n "$branch" ]] || die "refusing to publish from detached HEAD"

git remote get-url "$remote" >/dev/null 2>&1 \
  || die "unknown Git remote: $remote"

if [[ "$fetch_tags" -eq 1 ]]; then
  run git fetch --tags "$remote"
fi

latest_version="$(latest_local_version "$tag_prefix")"
if [[ -n "$explicit_version" ]]; then
  next_version="$explicit_version"
else
  next_version="$(bump_version "$latest_version" "$bump")"
fi
next_tag="${tag_prefix}${next_version}"

has_changes=0
if [[ -n "$(git status --porcelain)" ]]; then
  has_changes=1
fi

tag_exists=0
if git rev-parse -q --verify "refs/tags/$next_tag" >/dev/null; then
  tag_exists=1
  tag_commit="$(git rev-list -n 1 "$next_tag")"
  head_commit="$(git rev-parse HEAD)"
  if [[ "$tag_commit" != "$head_commit" ]]; then
    die "tag already exists on another commit: $next_tag"
  fi
  if [[ "$has_changes" -eq 1 ]]; then
    die "tag already exists on HEAD, but the working tree has changes: $next_tag"
  fi
  printf 'Local tag %s already exists on HEAD; resuming push.\n' "$next_tag"
fi

if [[ "$has_changes" -eq 1 ]]; then
  run git diff --check
  run git diff --cached --check
fi

if [[ "$run_tests" -eq 1 ]]; then
  if command -v uv >/dev/null 2>&1; then
    run uv run pytest
  else
    run python -m pytest
  fi
fi

if [[ "$has_changes" -eq 1 ]]; then
  run git add -A
  run git diff --cached --check
  if [[ "$dry_run" -eq 0 ]]; then
    git diff --cached --quiet && die "no staged changes after git add -A"
  fi
  run git commit -m "$commit_message"
elif [[ "$tag_exists" -eq 0 ]]; then
  printf 'No local changes; tagging current HEAD.\n'
fi

if [[ "$tag_exists" -eq 0 ]]; then
  run git tag -a "$next_tag" -m "Release $next_tag"
fi
run git push "$remote" "HEAD:$branch"
run git push "$remote" "$next_tag"

if [[ "$dry_run" -eq 1 ]]; then
  printf 'Dry run complete; next tag would be %s from branch %s.\n' \
    "$next_tag" "$branch"
else
  printf 'Published %s from branch %s.\n' "$next_tag" "$branch"
  printf 'GitHub Actions should publish the package from the tag push.\n'
fi
