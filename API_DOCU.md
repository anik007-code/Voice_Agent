# API Documentation

## POST /start-call
- **Input**: `{ "phone_number": str, "customer_name": str }`
- **Output**: `{ "call_id": str, "message": str, "first_message": str }`
- **Description**: Starts a new call and returns a call ID and initial greeting.

## POST /respond/{call_id}
- **Input**: `{ "message": str }`
- **Output**: `{ "call_id": str, "message": str, "first_message": null }`
- **Description**: Processes customer response and returns the agent’s reply.

## GET /conversation/{call_id}
- **Output**: `{ "call_id": str, "history": [{ "role": str, "message": str, "audio": str, "stage": str }] }`
- **Description**: Retrieves the conversation history for a given call ID.
