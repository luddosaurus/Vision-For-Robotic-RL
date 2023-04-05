print("here now")
from minimal_setup.subpackage.sub_file import MyClass

print("and here?")


def main():
    my_class = MyClass("Amelia")
    my_class.say_my_name()


main()
