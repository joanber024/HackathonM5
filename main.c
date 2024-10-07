#include <stdio.h>
#include "config.h"

int main(){
    readConfiguration("test-files/test1");
    printConfiguration();

    return 0;
}