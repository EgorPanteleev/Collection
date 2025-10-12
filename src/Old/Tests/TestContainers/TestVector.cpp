//
// Created by auser on 11/2/24.
//
#include <gtest/gtest.h>
#include <iostream>
#include <utility>
#include "Vector.h"
#define EPSILON 1e-5

struct TMP {
    TMP( int i, std::string s, double d ): i(i), s(std::move(s)), d(d) {}
    TMP( const TMP& other ) = default;
    int i;
    std::string s;
    double d;
};




TEST( TestVector, TestConstructors ) {
    Vector<TMP> vec1;
    EXPECT_EQ( vec1.size(), 0 );
    EXPECT_EQ( vec1.capacity(), 0 );
    EXPECT_EQ( vec1.data(), nullptr );

    Vector<TMP> vec2(5, TMP( 1, "str", 1.1 ) );
    EXPECT_EQ( vec2.size(), 5 );
    EXPECT_EQ( vec2.capacity(), 5 );
    EXPECT_NE( vec2.data(), nullptr );
    EXPECT_EQ( vec2[0].i, 1 );
    EXPECT_EQ( vec2[0].s, "str" );
    EXPECT_EQ( vec2[0].d, 1.1 );


    Vector<TMP> vec3 = vec2;
    EXPECT_EQ( vec3.size(), 5 );
    EXPECT_EQ( vec3.capacity(), 5 );
    EXPECT_NE( vec3.data(), nullptr );
    EXPECT_EQ( vec3[0].i, 1 );
    EXPECT_EQ( vec3[0].s, "str" );
    EXPECT_EQ( vec3[0].d, 1.1 );

    Vector<TMP> vec4 = std::move( vec2 );
    EXPECT_EQ( vec3.size(), 5 );
    EXPECT_EQ( vec3.capacity(), 5 );
    EXPECT_NE( vec3.data(), nullptr );
    EXPECT_EQ( vec3[0].i, 1 );
    EXPECT_EQ( vec3[0].s, "str" );
    EXPECT_EQ( vec3[0].d, 1.1 );

    EXPECT_EQ( vec2.size(), 0 );
    EXPECT_EQ( vec2.data(), nullptr );


}

TEST( TestVector, TestOperators ) {
    Vector<TMP> vec2(5, TMP( 1, "str", 1.1 ) );

    Vector<TMP> vec3;
    vec3 = vec2;
    EXPECT_EQ( vec3.size(), 5 );
    EXPECT_EQ( vec3.capacity(), 5 );
    EXPECT_NE( vec3.data(), nullptr );
    EXPECT_EQ( vec3[0].i, 1 );
    EXPECT_EQ( vec3[0].s, "str" );
    EXPECT_EQ( vec3[0].d, 1.1 );

    Vector<TMP> vec4;

    vec4 = std::move(vec2);
    EXPECT_EQ( vec3.size(), 5 );
    EXPECT_EQ( vec3.capacity(), 5 );
    EXPECT_NE( vec3.data(), nullptr );
    EXPECT_EQ( vec3[0].i, 1 );
    EXPECT_EQ( vec3[0].s, "str" );
    EXPECT_EQ( vec3[0].d, 1.1 );

    EXPECT_EQ( vec2.size(), 0 );
    EXPECT_EQ( vec2.data(), nullptr );
}

TEST( TestVector, TestMethods ) {
    Vector<TMP> vec;
    vec.reserve( 6 );
    EXPECT_EQ( vec.size(), 0 );
    EXPECT_EQ( vec.capacity(), 6 );
    EXPECT_NE( vec.data(), nullptr );

    vec.reserve( 5 );
    EXPECT_EQ( vec.size(), 0 );
    EXPECT_EQ( vec.capacity(), 6 );
    EXPECT_NE( vec.data(), nullptr );


    vec.resize(6, { 2, "str2", 2.2 });
    EXPECT_EQ( vec.size(), 6 );
    EXPECT_EQ( vec.capacity(), 6 );
    EXPECT_EQ( vec[1].i, 2 );
    EXPECT_EQ( vec[1].d, 2.2 );
    EXPECT_EQ( vec.front().i, 2 );
    EXPECT_EQ( vec.front().d, 2.2 );
    EXPECT_EQ( vec.back().i, 2 );
    EXPECT_EQ( vec.back().d, 2.2 );
    vec.resize(7, { 2, "str2", 2.2 });
    EXPECT_EQ( vec.size(), 7 );
    EXPECT_EQ( vec.capacity(), 7 );
    vec.resize(4, { 2, "str2", 2.2 });

    EXPECT_EQ( vec.size(), 4 );
    EXPECT_EQ( vec.capacity(), 7 );

    vec.push_back( { 3, "str3", 3.3 } );
    EXPECT_EQ( vec.size(), 5 );
    EXPECT_EQ( vec.capacity(), 7 );
    EXPECT_EQ( vec.back().i, 3 );
    EXPECT_EQ( vec.back().d, 3.3 );

    vec.pop_back();
    EXPECT_EQ( vec.size(), 4 );
    EXPECT_EQ( vec.capacity(), 7 );
    EXPECT_EQ( vec.back().i, 2 );
    EXPECT_EQ( vec.back().d, 2.2 );

    vec.emplace_back( 3, "str3", 3.3 );
    EXPECT_EQ( vec.size(), 5 );
    EXPECT_EQ( vec.capacity(), 7 );
    EXPECT_EQ( vec.back().i, 3 );
    EXPECT_EQ( vec.back().d, 3.3 );

}

TEST( TestVector, TestIterators ) {
    Vector<TMP> vec( 6, { 2, "str2", 2.2 } );
    int cnt = 0;
    for ( auto v: vec ) {
        ++cnt;
    }
    EXPECT_EQ( cnt, 6 );
    EXPECT_EQ( vec.end() - vec.begin(), 6 );
    Vector<TMP>::Iterator it = vec.begin();

    EXPECT_EQ( it->s, "str2" );
    EXPECT_EQ( (*it).d, 2.2 );
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
//    ::testing::GTEST_FLAG( color ) = "yes";
    return RUN_ALL_TESTS();
}
