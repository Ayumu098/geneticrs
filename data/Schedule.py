import torch
from datetime import datetime

# Time constants
DAYS_IN_WEEK = 7
HOURS_IN_DAY = 24
MINUTES_IN_HOUR = 60

DAYS = "MTWHFSU"


class Schedule:
    PERIOD = 5
    MINUTES_IN_WEEK = 10080
    PERIODS_IN_WEEK = MINUTES_IN_WEEK // PERIOD
    PERIODS_IN_DAY = PERIODS_IN_WEEK // DAYS_IN_WEEK
    PERIODS_IN_HOUR = PERIODS_IN_DAY // HOURS_IN_DAY

    DAY_OFFSET = dict(zip(DAYS, range(0, PERIODS_IN_WEEK, PERIODS_IN_DAY)))

    """Represents the schedule of a subject or course.
    """

    def __init__(self, source: str):
        """Loads the schedule from a string.

        Args:
            source (str): Text representing the schedule of a course.
            Format:
                `day` `time` `'AM' or'PM'`-`time` `'AM' or'PM'` *; ...
            Example:
                `F 4:05-6AM lec TBA; WTh 4-7PM lab TBA`
        """
        self._source = source

    @staticmethod
    def empty_schedule():
        """Returns an empty schedule."""
        return torch.zeros(Schedule.PERIODS_IN_WEEK)

    @staticmethod
    def military(time: str):
        """
        Args:
            time (`str`): Time format in varying formats:
                "x:xxAM/PM-x:xxAM/PM" (no change),
                "x-xAM/PM" (add AM/PM and :00 to the end),

        Returns:
                `str`: Time start in military format "xx:xx"
                `str`: Time end in military format "xx:xx"
        """
        start, end = time.split("-")

        # Add AM/PM if start and end use the same period
        # Example: 9-12PM -> 9PM-12PM
        if not any(period in start for period in ("AM", "PM")):
            start = start + end[-2:]

        # Format the time to be "x:xxAM/PM" (len=6) if it isn't already
        start = start[:-2] + ":00" + start[-2:] if ":" not in start else start
        end = end[:-2] + ":00" + end[-2:] if ":" not in end else end

        # Actual conversion after seperating
        start = datetime.strptime(start, "%I:%M%p").strftime("%H:%M")
        end = datetime.strptime(end, "%I:%M%p").strftime("%H:%M")

        return start, end

    @staticmethod
    def binary(day: str, start: str, end: str):
        """_summary_
        Args:
            day (`str`): Day of the week (in "MTWHFS", no Sunday).

            start (`str`): Start of a schedule in military time.

            end (`str`): End of a schedule in military time.

        Returns:
            `torch.tensor(PERIODS_IN_WEEK)`: Binary representation
            of the schedule. Wherein a toggled value represents
            occupancy of a period.
        """

        day_offset = Schedule.DAY_OFFSET[day]

        # Convert start into a single value
        # Representing the offset from the start of the week
        # in Schedule.Periods
        hours, minutes = start.split(":")
        hour_offset = int(hours) * Schedule.PERIODS_IN_HOUR
        minute_offset = int(minutes) // Schedule.PERIOD

        start = day_offset + hour_offset + minute_offset

        # Convert end into a single value
        # Representing the offset from the start of the week
        # in Schedule.Periods
        hours, minutes = end.split(":")
        hour_offset = int(hours) * Schedule.PERIODS_IN_HOUR
        minute_offset = int(minutes) // Schedule.PERIOD

        end = day_offset + hour_offset + minute_offset

        # Create a binary partition of the week
        # Wherein each period is represented by a 1 or 0
        # Example: [1, ..., 0] means the first period is occupied
        #          [0, ..., 1] means the last period is occupied
        binary_partition = Schedule.empty_schedule()
        binary_partition[range(start, end)] = 1

        return binary_partition

    @staticmethod
    def parse(source: str):
        """Convert a single text representation of a schedule into a list of
          subschedules (day, start, end) in military time.

        Args:
            source (str): Text representation of a schedule for a subject.

        Returns:
            list[list[str, str, str]]: List of subschedules for a subject.
        """

        # Split source into multiple subschedules
        schedules = source.split(";")

        # Extract the day and time from each schedule
        schedules = (schedule.split(" ")[:2] for schedule in schedules)

        # Replace every Thursday with H for convenience
        schedules = [(days.replace("Th", "H"), time) for days, time in schedules]

        # Replace every Sunday with U for convenience
        schedules = [(days.replace("Su", "U"), time) for days, time in schedules]

        # Split multiday schedules into seperate days
        schedules = [(day, time) for days, time in schedules for day in days]

        # Convert schedules into military time
        return [[day, *Schedule.military(time)] for day, time in schedules]

    @property
    def sub_schedules(self):
        """Returns the subschedules of the subject in military time.

        Returns:
            list[int, int, int]: [day, time_start, time_end]
        """
        return self.parse(self._source)

    @property
    def binary_format(self):
        """Return a binary representation of the subschedules."""
        schedule = Schedule.empty_schedule()

        for day, start, end in self.sub_schedules:
            schedule = schedule.add(Schedule.binary(day, start, end))

        return schedule
