class Person:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def set_name(self, new_name):
        self.name = new_name

    def set_age(self, new_age):
        self.age = new_age

    def set_gender(self, new_gender):
        self.gender = new_gender

    def __str__(self):
        return self.name + " " + str(self.age) + " " + self.gender


if __name__ == "__main__":
    person1 = Person("John", 55, "Male")

    print(person1.name, "is", person1.age, "old.")
    print(person1.name, "is", str.lower(person1.gender))
    person1.set_gender("Apache Helicoper")
    print(person1.name, "is", str.upper(person1.gender))

    person2 = Person("Jane,", 46, "Female")
    print(person2)
