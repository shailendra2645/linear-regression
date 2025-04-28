all: 
	g++ linear_regression.cpp helper.cpp -o build/main -Wall -Werror -lm -std=c++17