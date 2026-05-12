# P-SmokeNet frontend

## Frontend structure

P-SmokeNet/
          frontend/
                  index.html
                  styles.css
                  js/
                     main.js
                     state.js
                     dom.js
                     format.js
                     ui.js
                     api.js
                     router.js
                     chart.js
                     pages/
                           demo.js
                           videos.js
                           technical.js

## Frontend overview

The P-SmokeNet frontend is a lightweight dashboard for reviewing surgical smoke prediction results.

It has 3 main views:
- **Demo** - case review for prepared demo videos, with synced playback and smoke warning score over time
- **Demo Videos** — gallery of prepared backend served demo videos
- **Technical** — experiment summaries, run details, and comparison tables

The frontend connects to the backend through a configurable API base URL, which defaults to:

```text
http://localhost:8000