from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class TrainingStatusCallback(TrainerCallback):
    """Callback for printing model training status"""
    
    def __init__(self, print_interval=1000):
        """
        Args:
            print_interval: Print parameter statistics every N steps (0 means print only at the beginning)
        """
        self.print_interval = print_interval
        self.has_printed_initial = False
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.is_world_process_zero and not self.has_printed_initial:
            print("\n" + "="*100)
            print("[Training Start - Model Parameter Status Check]")
            print("="*100)
            self._print_model_status(model, verbose=True)
            self.has_printed_initial = True
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if self.print_interval > 0 and state.global_step % self.print_interval == 0:
            if state.is_world_process_zero:
                print(f"\n{'='*100}")
                print(f"[Step {state.global_step} - Parameter Statistics]")
                print(f"{'='*100}")
                self._print_model_status(model, verbose=False)
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.is_world_process_zero:
            epoch_num = int(state.epoch) if state.epoch is not None else 0
            print(f"\n{'='*100}")
            print(f"[Epoch {epoch_num} Start - Parameter Status]")
            print(f"{'='*100}")
            self._print_model_status(model, verbose=False)
    
    def _print_model_status(self, model, verbose=False):
        if model is None:
            print("  ⚠️ Model is None, cannot print status")
            return
        
        total_params = 0
        trainable_params = 0
        component_stats = {}
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            
            if 'ecg_tower' in name:
                component = 'ECG Tower'
            elif 'ecg_projector' in name:
                component = 'ECG Projector'
            elif 'visual' in name:
                if 'merger' in name or 'deepstack' in name:
                    component = 'Vision Aligner'
                else:
                    component = 'Vision Tower'
            elif 'language_model' in name or 'model.language_model' in name:
                if 'embed' in name:
                    component = 'LLM Embeddings'
                elif 'lm_head' in name:
                    component = 'LLM Head'
                else:
                    component = 'LLM Backbone'
            else:
                component = 'Other'
            
            if component not in component_stats:
                component_stats[component] = {
                    'total': 0,
                    'trainable': 0,
                    'params': []
                }
            
            component_stats[component]['total'] += num_params
            if param.requires_grad:
                component_stats[component]['trainable'] += num_params
            
            if verbose:
                component_stats[component]['params'].append({
                    'name': name,
                    'requires_grad': param.requires_grad,
                    'num_params': num_params
                })
        
        print(f"\nTotal params: {total_params:,} ({total_params/1e9:.2f}B) | "
              f"Trainable: {trainable_params:,} ({trainable_params/1e9:.2f}B) | "
              f"Ratio: {trainable_params/total_params*100:.1f}%")
        
        print(f"\n{'Component':<20} {'Total':>15} {'Trainable':>15} {'Ratio':>8} {'Status'}")
        print("-" * 75)
        
        for component in sorted(component_stats.keys()):
            stats = component_stats[component]
            total = stats['total']
            trainable = stats['trainable']
            ratio = trainable / total * 100 if total > 0 else 0
            
            if ratio == 0:
                status = "❄️Frozen"
            elif ratio == 100:
                status = "🔥Training"
            else:
                status = "⚡Partial"
            
            print(f"{component:<20} {total:>14,} {trainable:>14,} {ratio:>7.1f}% {status}")
        
        print(f"\n[ECG Components]")
        ecg_components = ['ECG Tower', 'ECG Projector']
        for comp in ecg_components:
            if comp in component_stats:
                stats = component_stats[comp]
                is_trainable = stats['trainable'] > 0
                status_icon = '✅' if is_trainable else '❄️'
                status_text = 'Training' if is_trainable else 'Frozen'
                print(f"  {status_icon} {comp:<18}: {status_text} ({stats['trainable']:,} / {stats['total']:,})")
            else:
                print(f"  ⚠️ {comp:<18}: Not found")
        
        if verbose and 'ECG Projector' in component_stats:
            print(f"\n[ECG Projector Detailed Parameters]")
            for p in component_stats['ECG Projector']['params']:
                marker = '✓' if p['requires_grad'] else '✗'
                grad_text = 'ON' if p['requires_grad'] else 'OFF'
                print(f"  {marker} {p['name']:<70} grad={grad_text}")
        
        print("")


training_status_callback = TrainingStatusCallback(print_interval=1000)

try:
    from swift.plugin import extra_callbacks
    extra_callbacks.append(training_status_callback)
    print("TrainingStatusCallback registered to swift.plugin.extra_callbacks")
except Exception as e:
    print(f"⚠️ Failed to register TrainingStatusCallback: {e}")
