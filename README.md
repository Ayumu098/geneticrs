# UPD CRS Genetic Algorithm Scheduling 

## Description

This is the official code implementation for a custom genetic algorithm system in determining an optimal enlistemnt under a simplified simulation of the University of the Philippines Diliman (UPD) Computer Registration System (CRS) that enlists subjects to students on a lottery system based on the demand and availability of a schedule.

This is submitted as the final requirement for ECE 197 -
Special Topics in Data Driven Control (Academic Year 2022-2023), under the supervision of Mr. John Audie Cabrera in the Electrical and Electronics Engineering Institute of the University of the Philippines Diliman.

## Intuition

The genetic algorithm determines options that improve the following parameters of the enlistment option(s):
- The probability of getting the type of subjects (calculated from demand and available slots)
- The balance of the probabilities (so a subject type isn't over prioritized)
- The minimization of the time overlap between subjects

## Dependencies
- Download [`Python`](https://www.python.org/downloads/) (at least version 3.10.11). We recommend using [Anaconda](https://www.anaconda.com/download).
- Download [`PyTorch`](https://pytorch.org/get-started/locally/) (Program can run on `CUDA` enabled GPU or `cpu`).
- Download [`matplotlib`](https://matplotlib.org/stable/users/installing/index.html)
- Download [`tqdm`](https://github.com/tqdm/tqdm#installation).

## Instructions

### Project Repository

Clone the GitHub repository using git

```console
git clone https://github.com/Ayumu098/geneticrs.git
```

Enter the project repository

```console
cd geneticrs
```

### Input File

Create a `.csv` file to add all the subjects that could be enlisted. The first row is the header rows, with columns `Class Name`, `Class Type`,`Available`, `Demand`, and `Schedule`.

#### Class Name
The `Class Name` column holds the name of the subjects. It doesn't need to be unique.

#### Class Type
The `Class Type` column holds the numbers that groups subjects together. 

For example, subject `A`, `B`, `C` can have the same class type `1` to signify that they are the same type. 

An example use case is grouping a set of subjects with the same name but different subjects or grouping completely different subjects for electives and general subjects.

#### Available Slots

The `Available` column holds the number of available slots for the subjects. Should be an integer.

#### Student Demands

The `Demand` column holds the current demand for the subjects. Should be an integer.

#### Schedule

The `Schedule` column holds the text format of the subjects' schedules. The format follows the UPD CRS format:

```
Days x1MM-y1MM *; Days x1MM-y1MM *; ...
```

- The Days can be a single day (example: `M`, `T`, `W`, `Th`, `F`, `S`, `Su`) or any non-repeating combination (example: `TTh`, `WF`, `MTWThF`, etc.).
- The time `xMM-yMM` consists of the start and end time in 12-hour formats. The `AM/PM` of the start time is ommitted if it's similar to the end time.
- The `*` are succeeding texts that will be ignored. These are usually the `LEC/LAB`, and additional notes.
- Some schedules can have multiple day-times, seperated by a semicolon `;`.

Examples of valid schedules are listed below:

```
S 9AM-12PM lab TBA; WF 10-11AM lec TBA
M 10AM-1PM lec TBA
MTWThF 9-11AM lec TBA(CD)
MTWThF 8AM-5PM lec TBA(AIT)
```

Simply copy paste the schedule from the UPD CRS (including the `LEC/LAB` and instructor as they will be removed automatically).

### Running Algorithm

Run the following command to use the default configurations of the genetic algorithm in determining the optimal enlistment. Replace `CSV_FILE_NAME.csv` with the filename of the `.csv` file from earlier.

```console
python solution.py --input-file=CSV_FILE_NAME.csv
```

## Configurations

There are many configurations to set in `solution.py`, here are some of them. Note that they can be cascaded. For more information, look into argeparse.

```console
python solution.py --setting1=value1 --setting2=value2 --setting3=value3
```

### Program Settings

`--device`
Set which device to use for `PyTorch` calculations. Defaults to `CUDA` if enabled in GPU; Else, uses the `cpu`.

```console
python solution.py --device=text
```

`--plot`
Shows the fitness over generations and the partition of selectoin, crossover, and mutations over generations as well if enabled.

```console
python solution.py --plot
```

### Algorithm Settings

`--generations`
This is the number of iterations used by the algorithm. Higher values will allow the algorithm to find better solutions but it will use more time.

```console
python solution.py --generations=number
```

`--gene-size`
The number of subjects to be enlisted at a time.

```console
python solution.py --gene-size=number
```

`population-size`
The number of options to consider. Lower if computation is too slow. However, don't set it to a very low number.
```console
python solution.py --population-size=number
```

### Probabilities

Note that the `mutation-probability` and `crossover-probability` combined shouldn't be more than `1.0`. Ideally, their sum should be less than `1.0` for stability.

`--mutation-probability`
A higher mutation probability will cause the algorithm to try more unconventional solutions. Increase this slightly if the fitness doesn't increase too much over generations.

```console
python solution.py --mutation-probability=decimal
```

`--crossover-probability`
A higher crossover probability will cause the algorithm to try a solution that uses pieces from the best solution. Increase this slightly if the fitness doesn't increase too much over generations.

```console
python solution.py --crossover-probability=decimal
```

### Weights

`--weight-probability`
Determines the emphasis on increasing the chances of getting each type of subject. Defaults to 1 (equal to other weights). Try increasing this if the number of enlistments in the results is low.

```console
python solution.py --weight-probability=number
```

`--weight-balance`
Determines the emphasis on balancing the probability of getting the subjects. Defaults to 1. A high weight means the optimizer tries to ensure no specific subject type is prioritized. Set to zero if balance doesn't matter.

```console
python solution.py --weight-balance=number
```

`--weight-overlap`
Determines the emphasis on penalizing overlaps between subject types (in terms of time). Defaults to 1. Increase to prevent collisions.

```console
python solution.py --weight-overlap=number
```

## Randomizer

The code will actually output the same result everytime despite being probabilistic. To change the result, change the seed. Of course, using the same seed results the same results.

`--seed`
```console
python solution.py --seed=number
```

To choose a completely random seed, use the following flag. Any use of `--seed` will be ignored.

`--no-seed`
```console
python solution.py --no-seed
```

## Scopes and Limitations
The enlistment system does not account for the user's preference on subject for a given type, the handlers, the time period, the location, or the prerequisite and corequisites of the subjects. The system also does not account for priority enlistment.

## Miscellaneous
For any concern with the project, kindly raise an issue. Feel free to submit a pull request.