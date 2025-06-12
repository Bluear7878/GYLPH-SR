#include <iostream>
#include <nlohmann/json.hpp>
#include <string_view>

using namespace std::string_view_literals;
using json = nlohmann::json;

int main()
{
    // create a JSON object
    json j_object = {{"one", 1}, {"two", 2}};

    // call erase()
    auto count_one = j_object.erase("one" sv);
    auto count_three = j_object.erase("three" sv);

    // print values
    std::cout << j_object << '\n';
    std::cout << count_one << " " << count_three << '\n';
}
