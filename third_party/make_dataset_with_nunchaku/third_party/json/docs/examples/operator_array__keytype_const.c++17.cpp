#include <iostream>
#include <nlohmann/json.hpp>
#include <string_view>

using namespace std::string_view_literals;
using json = nlohmann::json;

int main()
{
    // create a JSON object
    const json object =
        {
            {"one", 1},
            {"two", 2},
            {"three", 2.9}};

    // output element with key "two"
    std::cout << object["two" sv] << '\n';
}
