.PHONY: compress-central-hub dry-run-central-hub

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
