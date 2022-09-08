run_train:
	python -c 'from job_prepr_model.interface.main import train; train()'

run_train_sample:
	python -c 'from job_prepr_model.interface.main import train; train(sample=0.2)'

run_validate:
	python -c 'from job_prepr_model.interface.main import validate; validate()'
