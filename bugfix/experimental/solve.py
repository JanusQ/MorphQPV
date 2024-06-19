import jax.numpy as jnp
from jax import value_and_grad, jit, grad
from tqdm import tqdm
from jax import random
from jax.example_libraries.optimizers import adam
class SgdOptimizer:
    def __init__(self,step_size=0.01,steps=5000):
        self.step_size = step_size
        self.steps = steps
    def optimize_params(self,sampled_probs,target_probs,dims):
        """ 优化器
        Args:
            inputs: 输入的量子态
            outputs: 重建的量子态
            real_target: 真实的量子态
            target: 重建的目标，input or output
        return:
            bast_params: 优化后的参数
        """
        parms= random.normal(random.PRNGKey(0), (dims,))
        min_iter = 0
        opt_init, opt_update, get_params = adam(self.step_size)
        min_cost = 1e10
        last_cost = min_cost
        opt_state = opt_init(parms)
        def build_probs(parms,sampled_probs):
            return jnp.sum(jnp.array([parm**2*sampled_prob for parm,sampled_prob in zip(parms,sampled_probs)]),axis=0)
        def sgd_cost_function(parms,sampled_probs,target_probs):
            return jnp.mean(jnp.abs(build_probs(parms,sampled_probs)-target_probs))

        with tqdm(range(self.steps)) as pbar:
            for i in pbar:
                params = get_params(opt_state)
                cost, grads = value_and_grad(sgd_cost_function)(params,sampled_probs, target_probs)
                opt_state=  opt_update(i, grads, opt_state)
                if cost < min_cost:
                    min_cost = cost
                    bast_params = params
                    pbar.set_description(f'sgd optimizing ')
                    #设置进度条右边显示的信息
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                if jnp.abs(min_cost-last_cost) < 1e-5:
                    min_iter+=1
                else:
                    min_iter = 0
                # 当连续50次迭代损失函数变化小于1e-5时，认为收敛
                if min_iter > 50:
                    pbar.set_description(f'sgd optimizing converge')
                    pbar.set_postfix(loss=cost, min_loss=min_cost)
                    break
                last_cost = min_cost
        return bast_params