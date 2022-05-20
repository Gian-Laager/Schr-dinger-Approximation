#include "pch.h"

TEST(test, test)
{
    ASSERT_EQ(add(1, 2), 1 + 2);
}

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
