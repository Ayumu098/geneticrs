from .Schedule import Schedule, DAYS
from datetime import datetime
from random import randint, choice


def test_standard_military():
    """
    Test military time conversion on x:xxAM/PM-x:xxAM/PM" format.
    """

    # Generate start time

    hours = randint(1, 12)
    minutes = randint(0, 59)
    period = choice(["AM", "PM"])

    test_start = f"{hours}:{minutes}{period}"

    # Generate end time

    hours = randint(1, 12)
    minutes = randint(0, 59)
    period = choice(["AM", "PM"])

    test_end = f"{hours}:{minutes}{period}"

    # True military time formatting

    true_start = datetime.strptime(test_start, "%I:%M%p").strftime("%H:%M")
    true_end = datetime.strptime(test_end, "%I:%M%p").strftime("%H:%M")

    # Test Schedule military

    start, end = Schedule.military(f"{test_start}-{test_end}")

    assert true_start == start
    assert true_end == end


def test_simplified_military():
    """
    Test military time conversion on "x-xAM/PM" format.
    """

    # Generate start and end time for testing

    minutes = 0
    period = choice(["AM", "PM"])

    initial_hours = randint(1, 12)
    final_hours = randint(1, 12)

    test_start = f"{initial_hours}:{minutes}{period}"
    test_end = f"{final_hours}:{minutes}{period}"

    # True military time formatting

    true_start = datetime.strptime(test_start, "%I:%M%p").strftime("%H:%M")
    true_end = datetime.strptime(test_end, "%I:%M%p").strftime("%H:%M")

    # Test Schedule military

    start, end = Schedule.military(f"{initial_hours}-{test_end}")

    assert true_start == start
    assert true_end == end


def test_parse():
    for _ in range(randint(1, 100)):
        days = "".join(choice(DAYS) for _ in range(randint(1, 10)))

        # Generate start time
        start_period = choice(("AM", "PM"))
        start_minutes = randint(0, 59)
        start_hours = randint(1, 12)
        start_time = f"{start_hours}:{start_minutes}{start_period}"

        # Generate end time
        end_period = choice(("AM", "PM"))
        end_minutes = randint(0, 59)
        end_hours = randint(1, 12)
        end_time = f"{end_hours}:{end_minutes}{end_period}"

        # Generate true parsed schedule
        true_start = datetime.strptime(start_time, "%I:%M%p").strftime("%H:%M")
        true_end = datetime.strptime(end_time, "%I:%M%p").strftime("%H:%M")
        true_schedules = [[day, true_start, true_end] for day in days]

        # Use Schedule.parse to generate test schedule
        test_schedule = f"{days} {start_time}-{end_time}"
        schedule = Schedule(test_schedule)

        assert schedule.sub_schedules == true_schedules


def test_binary_initial():
    initial_schedule = Schedule("M 12:00AM-12:05AM")
    true_initial_schedule = Schedule.empty_schedule()
    true_initial_schedule[0] = 1

    assert initial_schedule.binary_format.equal(true_initial_schedule)


def test_binary_final():
    final_schedule = Schedule("Su 11:50PM-11:55PM")
    true_final_schedule = Schedule.empty_schedule()
    true_final_schedule[-2] = 1

    assert final_schedule.binary_format.equal(true_final_schedule)
