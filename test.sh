mkdir -p dist &&
g++ -fsanitize=address -O2 -g --std=c++2a test.cpp -o dist/digit &&
time ./dist/digit
