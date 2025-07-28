dev, d:
	npm run dev
start, s:
	npm run build
	npm run start
test, t:
	npm run test
lint:
	npm run lint
format:
	npm run format
stress:
	npm run stress

docker:
	@if [ -n "$(wait)" ]; then \
		./run_embedding_service_and_test.sh -w $(wait); \
	else \
		./run_embedding_service_and_test.sh; \
	fi
