# PathSnap API rehearsal

## Integrate the team’s prediction pipeline

Run the configured app as described in the [two-laptop setup](../README.md#run-on-two-laptops).
It loads the exported classifier, runs Viterbi decoding, and scores the result
using the reserved reference route on the API laptop.

- The predictor receives a `Trip` containing the GPS track and road network.
  Its `route` field is always `None`. Return one arc ID per observation, with
  `-1` for observations without a prediction.
- The scorer receives the selected trip directory and the predicted arc IDs
  after inference completes. Return only aggregate scores.
- The UI reads trip IDs and point counts from `GET /trips` and calls
  `POST /predict/{trip_id}`. A successful prediction contains the trip ID, GPS
  points and route segments in latitude/longitude order, scores, and runtime.

## Rehearse on the evaluation LAN

1. On laptop 1, start the team-integrated FastAPI app with Uvicorn on the
   evaluation LAN. Use the evaluator-supplied trip directory for the official
   rehearsal. Use the student-test directory during development.
2. On laptop 2, serve Chiranthan’s UI and enter laptop 1’s address and API port.
   Set the API’s allowed origin to laptop 2’s UI URL, including the port.
3. Confirm that ten trips are listed. Select each trip and confirm that its
   route and metrics display and that laptop 1 shows a matching HTTP 200
   request log. Capture the UI and API terminal for the team report.

The API tests verify request routing, logging, allowed origins, and the
separation between predictions and reference data using a synthetic trip.
