# Assignment Lab 3

## Requirements

The code was written for Python 3.6.6 with Jupyter Notebook. Some requirements can be installed using `pip` (or `pip3` if you have separate pip for Python 3):

```
pip install mmh3
pip install hmmlearn
pip install scikit_learn
pip install matplotlib
pip install pandas
pip install numpy
```

## The Dataset & Preloaded Files
For the dataset, please download the following:

1. [Dataset for Sampling & Sketching](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/capture20110811.pcap.netflow.labeled)
2. [Dataset for Discretization & Profiling](https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/capture20110818.pcap.netflow.labeled) (rename this one to `capture-scenario10.pcap.netflow.labeled` to avoid conflict)

and put them in the `dataset` folder.

There area also some files in the `output` which should allow you to run the code more quickly, skipping time consuming process.

## Running the code
You would need to run in in Jupyter, so run jupyter:

```
jupyter notebook
```

and open each of the following .ipynb files for the respective task:

1. [sampling_task.ipynb](https://github.com/enreina/cs4035-lab/blob/master/lab3/sampling_task.ipynb)
2. [Sketching_Task.ipynb](https://github.com/enreina/cs4035-lab/blob/master/lab3/Sketching_Task.ipynb)
3. [flow_discretization_task.ipynb](https://github.com/enreina/cs4035-lab/blob/master/lab3/flow_discretization_task.ipynb)
4. [profiling.ipynb](https://github.com/enreina/cs4035-lab/blob/master/lab3/profiling.ipynb)

