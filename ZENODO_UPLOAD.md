# Zenodo release checklist

This repository is prepared for archiving through the GitHub-Zenodo integration.

## Before creating the GitHub release

```bash
git status
git ls-files | grep -E '(^figures/|results/.*/fits/|\.(png|jpg|svg|pdf|docx|zip|tif|tiff|eps)$|^~\$)'
git log --format='%an <%ae>' -5
```

The second command should print nothing. The commit history should identify Simon Salaj only.

Version 1.0.0 is already archived. Version 1.1.0 is the revision-analysis update.

After committing and pushing the update, create the release tag:

```bash
git tag -a v1.1.0 -m "Version 1.1.0"
git push origin main --tags
```

## GitHub-Zenodo integration

1. Sign in to Zenodo using the GitHub account `simonator420`.
2. From the Zenodo profile menu open **GitHub**, click **Sync now**, and confirm that `simonator420/hift-between-participant-variability` remains enabled. Do this before publishing the GitHub release.
3. Create the release:

   ```bash
   gh release create v1.1.0 \
     --repo simonator420/hift-between-participant-variability \
     --title "HIFT Between-Participant Variability, v1.1.0" \
     --notes-file RELEASE_NOTES_v1.1.0.md
   ```

4. Wait for Zenodo to archive the release.
5. Check the new Zenodo version before using its DOI: creator `Simon Salaj` only, resource type `Software`, version `1.1.0`, license `MIT`, open access, and the correct GitHub link.
6. Use the new version DOI when citing the exact revised archive. The concept DOI can be used when a citation should always resolve to the newest version.

Do not use the GitHub-generated source ZIP for manual Zenodo upload if it omits required files. For a manual deposit, create a clean archive from tracked files with `git archive`.
