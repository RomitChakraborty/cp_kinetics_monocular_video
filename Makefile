.PHONY: test montage montage_py

test:
	./tests/test_smoke.sh

montage:
	./scripts/montage.sh -l _tmp/list.txt -o _out/concat.mp4

montage_py:
	./scripts/montage.py -l _tmp/list.txt -o _out/concat.mp4
