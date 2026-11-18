# rtool demo video

- `rtool-demo.mp4`: 5 min 21 s, 1440×900, 39 MB (sharper; share by link)
- `rtool-demo-email.mp4`: the same at a lower bitrate, 17 MB (fits an email; text softens for about 3 s after big screen changes)

Both are recorded from the web UI in demo mode (simulated papers, model and cluster); waits are cut out.

## Re-record

1. Start a demo server with a neutral runs folder (the folder path is shown on the Datasets & models page):

       rm -rf /tmp/rtool-demo && mkdir -p /tmp/rtool-demo/runs
       LOGNAME=researcher USER=researcher PYTHONPATH=src python -c 'import sys; from research_tool.cli.main import app; sys.argv[0]="rtool"; app()' \
         ui --mode local --demo --runs-root /tmp/rtool-demo/runs --port 8811 --token-file /tmp/rtool-demo/token

2. Record (`node_modules` links to a frontend install with Playwright; uses the installed Google Chrome), with nothing heavy running:

       DEMO_TOKEN=/tmp/rtool-demo/token npx playwright test tour.rec.ts

3. Encode (macOS AVFoundation, no ffmpeg) and check stills in Chrome:

       swift encode.swift out/tour-frames rtool-demo.mp4
       BITRATE=420000 swift encode.swift out/tour-frames rtool-demo-email.mp4
       mkdir -p out/check && VIDEO=rtool-demo.mp4 AT=10,60,120 npx playwright test check.rec.ts

Files: `tour.rec.ts` (the script and captions), `recorder.ts` (screencast capture, captions, cursor, cut waits),
`encode.swift`, `check.rec.ts`, `explore.rec.ts` (dumps every page of the staged demo runs), `stills.swift` (unused:
AVFoundation could not read frames back on this Mac; `check.rec.ts` does that in Chrome).
