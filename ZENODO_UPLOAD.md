# Zenodo release checklist

This repository is prepared for archiving through the GitHub-Zenodo integration.

## Before creating the GitHub release

```bash
git status
git ls-files | grep -E '(^figures/|results/.*/fits/|\.(png|jpg|svg|pdf|docx|zip|tif|tiff|eps)$|^~\$)'
git log --format='%an <%ae>' -5
```

The second command should print nothing. The commit history should identify Simon Salaj only.

Create and push the release tag:

```bash
git tag -a v1.0.0 -m "Version 1.0.0"
git push origin main --tags
```

## GitHub-Zenodo integration

1. Sign in to Zenodo using the GitHub account `simonator420`.
2. From the Zenodo profile menu open **GitHub**, click **Sync now**, find `simonator420/hift-between-participant-variability`, and enable it.
3. Only after the repository is enabled, create the release:

   ```bash
   gh release create v1.0.0 \
     --repo simonator420/hift-between-participant-variability \
     --title "Version 1.0.0" \
     --notes "First archived reproducibility release."
   ```

4. Wait for Zenodo to archive the release.
5. Check the Zenodo record before using the DOI: creator `Simon Salaj` only, resource type `Software`, version `1.0.0`, license `MIT`, and the correct GitHub link.
6. Add the version DOI to the manuscript Data Availability Statement and repository citation.

Do not use the GitHub-generated source ZIP for manual Zenodo upload if it omits required files. For a manual deposit, create a clean archive from tracked files with `git archive`.
