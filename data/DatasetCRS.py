from torch.utils.data import IterableDataset
from Schedule import Schedule
from pandas import read_csv


class DatasetCRS(IterableDataset):
    COLUMNS = [
        "Class Name",
        "Class Type",
        "Available",
        "Demand",
        "Schedule",
    ]

    def __init__(self, csv_source):
        self._length = 0
        self._data = self.parse_csv(csv_source)

    def parse_csv(self, csv_source):
        """Parses the csv file and returns a list of courses.
        CSV contains the following columns:
            Class Name: String
            Class Type: Number coded
            Available: Integer
            Demand: Integer
            Schedule: String (Follow the format expected in `Schedule` class)

        Args:
            csv_source (`str`): Path to the csv file.
        """

        # Read the csv file
        frames = read_csv(csv_source, header=0, usecols=DatasetCRS.COLUMNS)
        self._len = frames.shape[0]

        class_names = frames["Class Name"]
        class_types = frames["Class Type"]

        probabilities = (
            available / demand
            for available, demand in zip(frames["Available"], frames["Demand"])
        )

        schedules = (
            Schedule(schedule).binary_format for schedule in frames["Schedule"]
        )

        return zip(class_names, class_types, probabilities, schedules)

    def __len__(self):
        return self._len

    def __repr__(self):
        return f"DatasetCRS({self._len})"

    def __str__(self):
        return f"DatasetCRS({self._len})"

    def __iter__(self):
        return iter(self._data)

    def __next__(self):
        return next(self._data)
