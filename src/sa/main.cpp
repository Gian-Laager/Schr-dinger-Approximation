#include "pch.h"

int add(int a, int b)
{
    return a + b;
}

#ifdef SA_LIB
namespace sa
{
#endif

int main()
{
    std::cout << "1 + 2 = " << add(1, 2) << std::endl;
    return 0;
}

#ifdef SA_LIB
}
#endif