#include <gtest/gtest.h>

class Bis
{
public:
    bool Even(int n)
    {
        if (n % 2 == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    };

    bool Suc(bool bSuc)
    {
        return bSuc;
    }
};

class CombineTest :
    public ::testing::TestWithParam< ::testing::tuple<bool, int> >
{
protected:
	bool checkData() {
        int n = ::testing::get<1>(GetParam());      // <1> 1st parameter in the tuple
		bool suc = ::testing::get<0>(GetParam());   // <0> 0th parameter in the tuple
		return bis.Suc(suc) && bis.Even(n) || !bis.Suc(suc) && !bis.Even(n);
	}

private:
	Bis bis;
};

TEST_P(CombineTest, Test) {
	EXPECT_TRUE(checkData());
}

INSTANTIATE_TEST_CASE_P(TestBisValuesCombine1, CombineTest, ::testing::Combine(::testing::Bool(), ::testing::Values(0, 1, 2, 3)));
INSTANTIATE_TEST_CASE_P(TestBisValuesCombine, CombineTest, ::testing::Combine(::testing::Values(false, true), ::testing::Values(0, 1, 2, 3)));
//::testing::Bool() 是指遍历所有的 boolean 值
//::testing::Combine(A,B,C,...) 是由A x B x C x ... 构成的一个笛卡尔积
