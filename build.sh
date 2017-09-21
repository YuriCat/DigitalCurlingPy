CXX="g++"
CXXFLAGS="-std=c++14 -Wall -O3 -flto -DRULE_ERROR_GAT"
LIBRARIES="lib/Box2D/Box2D/Build/gmake/bin/Release/libBox2D.a"
INCLUDES="-Ilib/Box2D/Box2D/ -Ilib/pybind11/include/ -Ilib/DigitalCurling/"
PYTHON_VERSION="3"
${CXX} ${CXXFLAGS} -shared -fPIC ${INCLUDES} `python${PYTHON_VERSION}-config --cflags --ldflags` cpp/dccpp.cpp lib/DigitalCurling/CurlingSimulator.cpp ${LIBRARIES} -o python/dccpp.so