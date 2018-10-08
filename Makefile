
train:
	echo "no yet"


predict:
	python3 machine.py --xyz test/MULTI.xyz --model test/model.json --alpha test/alphas.npy --training test/representations.npy


requirments:
	pip install -r requirements.txt


PYTHON=python3

qml:
	${PYTHON} setup.py build
	cd build
	# cd lib.....
	# cp -r qml to any folder you want to use it
