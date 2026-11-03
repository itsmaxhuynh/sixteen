CXX ?= g++
CXXFLAGS ?= -std=c++20 -pthread -O3 -march=native -g -Wall

all: sixteen

sixteen: sixteen.cpp *.h
	$(CXX) $(CXXFLAGS) -o $@ sixteen.cpp

test: tests.cpp *.h
	$(CXX) $(CXXFLAGS) -o tests tests.cpp && ./tests

clean:
	rm -f sixteen tests

.PHONY: all clean test
