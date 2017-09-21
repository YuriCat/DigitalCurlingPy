CXX       = g++
CXXFLAGS  = -std=c++14 -Wall
LIBRARIES = lib/Box2D/Box2D/Build/gmake/bin/Release/libBox2D.a
INCLUDES  = -Ilib/Box2D/Box2D/ -Ilib/pybind11/include/ -Ilib/DigitalCurling/
PYTHON    = python3

ifeq ($(TARGET),debug)
	CXXFLAGS += -O0 -g -ggdb -D_GLIBCXX_DEBUG
endif
ifeq ($(TARGET),default)
	CXXFLAGS += -O3
endif

default debug:
	$(MAKE) TARGET=$@ dccpp

clean:
	rm dccpp

dccpp :
	$(CXX) $(CXXFLAGS) -shared -fPIC $(INCLUDES) $($(PYTHON)-config --cflags --ldflags) cpp/dccpp.cpp lib/DigitalCurling/CurlingSimulator.cpp $(LIBRARIES) -o python/dccpp
