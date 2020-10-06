#include <iostream>

int mu(int *x, int *stop){
//     _________________________
//    < multiple list of number >
//     -------------------------
//            \   ^__^
//             \  (oo)\_______
//                (__)\       )\/\
//                    ||----w |
//                    ||     ||
    if (x - stop == 0) return *x;
    return *x * mu(++x, stop);
}

using namespace std;

int main(){
    int x[] = {2, 3, 5, 6};
    int *stop = &x[3];
    cout << mu(x, stop);
    return 0;
}
