#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string_view>

using namespace std::string_view_literals;
using json = nlohmann::json;

int main()
{
    // create a JSON object
    json object =
        {
            {"one", 1},
            {"two", 2},
            {"three", 2.9}};

    // output element with key "two"
    std::cout << object["two" sv] << "\n\n";

    // change element with key "three"
    object["three" sv] = 3;

    // output changed array
    std::cout << std::setw(4) << object << "\n\n";

    // mention nonexisting key
    object["four" sv];

    // write to nonexisting key
    object["five" sv]["really" sv]["nested" sv] = true;

    // output changed object
    std::cout << std::setw(4) << object << '\n';
}
