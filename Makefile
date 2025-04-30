.PHONY: compress-central-hub dry-run-central-hub send_request


send_request:
	@if [ -z "$(URL)" ] || [ -z "$(DATA)" ]; then \
		echo "Usage: make send_request URL=<url> DATA='<json>'"; \
		exit 1; \
	fi
	curl -X POST "$(URL)" \
	-H "Content-Type: application/json" \
	-d '$(DATA)'




## Repomix Compression

# Pass compression level
compress-1:
	./compress-central-hub.sh 1
compress-2:
	./compress-central-hub.sh 2
compress-3:
	./compress-central-hub.sh 3
compress-4:
	./compress-central-hub.sh 4
compress-5:
	./compress-central-hub.sh 5

# Dry runs
dry-run-central-hub:
	./compress-central-hub.sh 2 --dry-run

dry-run-central-hub-max:
	./compress-central-hub.sh 5 --dry-run

# Help
compress-help:
	./compress-central-hub.sh --help


