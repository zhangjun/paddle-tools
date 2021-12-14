import paddle.fluid as fluid
import paddle
import numpy 

# prepare data
train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32')
y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32') 

paddle.enable_static()

# define network
x = fluid.layers.data(name="x",shape=[1],dtype='float32') 
y = fluid.layers.data(name="y",shape=[1],dtype='float32') 
y_predict = fluid.layers.fc(input=x,size=1,act=None)

# loss
cost = fluid.layers.square_error_cost(input=y_predict,label=y) 
avg_cost = fluid.layers.mean(cost) 

# optimizer
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost) 

# param init
cpu = fluid.core.CPUPlace() 
exe = fluid.Executor(cpu) 
exe.run(fluid.default_startup_program()) 

## training，100 iter
for i in range(100): 
    outs = exe.run( feed={'x':train_data,'y':y_true}, fetch_list=[y_predict.name,avg_cost.name]) 
    # train result 
    print(outs)

model_dir = "./fc"
fluid.io.save_inference_model(model_dir, ["x"], [y_predict], exe)
