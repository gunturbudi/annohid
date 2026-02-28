# AnnoHID

Medical text annotation platform with NER support, LLM-assisted labeling, and multi-annotator management.

## Running with Docker

```bash
docker compose up --build -d
```

Open **http://localhost:8810** once both containers are ready (~30s).

To stop:

```bash
docker compose down
```

To stop and remove all data:

```bash
docker compose down -v
```

## Demo Credentials

| Role      | Username       | Password   |
|-----------|----------------|------------|
| Admin     | administrator  | anohid123  |
| Annotator | annotator_22   | Rekmed123  |
