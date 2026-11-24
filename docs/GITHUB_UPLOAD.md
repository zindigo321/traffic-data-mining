# Publishing the Anonymous Copy to GitHub

This copy was prepared with anonymized filenames and text, Office and PDF author properties, and log identity fields. Keep earlier, non-anonymized coursework bundles, ZIPs, and decks out of the public repository and Releases. See `PrivacyReview.json` for the review.

## Create a Repository

Open <https://github.com/new> in your own account and create an empty public repository, for example `traffic-data-mining-course`. The local archive already contains its README and ignore rules.

## Local Checks and Initial Commit

The original public copy was prepared with `main` and the repository-local anonymous identity `Course Archive <archive@example.invalid>`.

```powershell
Set-Location -LiteralPath 'path/to/TrafficDataMining_GitHub'
git config --local user.name
git config --local user.email
git add .
git status --short
git diff --cached --stat
git commit -m "Archive anonymized traffic data mining coursework"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/traffic-data-mining-course.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` and the repository name as needed. If Git opens a browser on first push, sign in there. Keep passwords and tokens out of code, remote URLs, and chats. Repository ownership is visible on GitHub independently of the anonymous commit identity.

## Subsequent Updates

Review text, file properties, screenshots, bylines, and student IDs before adding files. Use the repository's anonymous commit identity. Review new run logs for local paths before publishing.

```powershell
git status
git add assignments README.md
git diff --cached --stat
git commit -m "Update anonymized coursework"
git push
```

The largest file in the original public copy was about 22.5 MB and fit in ordinary Git. Keep the original course archive, non-anonymized final ZIP, individual trajectories, and formal named coursework submissions in private storage.

## References

- [Add locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [Large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Set a commit email address](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)

The original preparation step covered local anonymization only. Repository creation, commit, and push followed separately.
