# Online Feature Selection Ensembles 

The code provided here implements online feature selection with bagging and boosting. We use Oza's implementation for online bagging and boosting with some modification to implement the ensemble feature selection component. 

# A Note Before Getting Started

If you are planning on using the experiment and plotting scripts that it is expected that your directories are setup in a very specific way. Create three directories in the root folder of the repository. 

```bash
mkdir eps/
mkdir pdf/
mkdir mat/  
```

Then run `run_experiments.m`. *Warning*: You'll need Matlab's distributed computing toolbox and the current implementation will open 25 workers. The plots and tables can be obtained by running parts of `plot_experiments.m`.  

# License 

MIT (See `LICENSE`)

# Contact 

* [Gregory Ditzler](http://gregoryditzler.com) (<gregory.ditzler@gmail.com>)`


