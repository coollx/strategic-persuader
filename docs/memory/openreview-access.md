---
name: openreview-access
description: OpenReview blocks anonymous access from this box (Cloudflare Turnstile); use the researcher's token at ~/.openreview_token with the api2 REST API
metadata:
  type: reference
---

openreview.net and api2.openreview.net answer anonymous requests from this machine with a Cloudflare Turnstile challenge (HTTP 403 `ChallengeRequiredError`); headless browsers cannot pass it (the widget refuses to render), and WebFetch/curl only see the challenge page. Authenticated requests skip the wall entirely.

**Why:** Discovered 2026-08-11 during /ref add for GTAlign; hours were spent on browser workarounds before the authenticated path proved to be the fix.

**How to apply:** Read the token from `~/.openreview_token` (created by the researcher; never print its value) and call e.g. `curl -H "Authorization: Bearer $(cat ~/.openreview_token)" "https://api2.openreview.net/notes?forum=<id>"`. If the file is missing or expired, ask the researcher to run `python3 scratch/or_login.py` themselves — never handle their password. The login endpoint itself is challenge-exempt, and `api2.openreview.net/notes/search?term=...` sometimes answers anonymously (good for venue-id/decision lookups).
