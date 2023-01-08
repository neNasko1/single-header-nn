mkdir -p dist &&
g++ -fsanitize=address -O2 -g --std=c++2a digit_recognition.cpp -o dist/digit &&
./dist/digit
