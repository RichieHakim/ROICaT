import torch
import torchvision

#### Paste ModelTackOn Definition from above Here...
### Define ModelTackOn
class ModelTackOn(torch.nn.Module):
    def __init__(
        self,
        base_model,
        un_modified_model,
        data_dim=(1,3,36,36),
        pre_head_fc_sizes=[100],
        post_head_fc_sizes=[100],
        classifier_fc_sizes=None,
        nonlinearity='relu',
        kwargs_nonlinearity={},
    ):
        """
        Initialize Model
        """
        super(ModelTackOn, self).__init__()
        self.base_model = base_model
        final_base_layer = list(un_modified_model.children())[-1]
        # final_base_layer = list(list(model.children())[-1].children())[-1] print(final_base_layer)
                
        self.data_dim = data_dim
        self.pre_head_fc_lst = []
        self.post_head_fc_lst = []
        self.classifier_fc_lst = []
            
        self.nonlinearity = nonlinearity
        self.kwargs_nonlinearity = kwargs_nonlinearity
        self.init_prehead(final_base_layer, pre_head_fc_sizes)
        self.init_posthead(pre_head_fc_sizes[-1], post_head_fc_sizes)
        if classifier_fc_sizes is not None:
            self.init_classifier(pre_head_fc_sizes[-1], classifier_fc_sizes)
            
            
    def init_prehead(self, prv_layer, pre_head_fc_sizes):
        for i, pre_head_fc in enumerate(pre_head_fc_sizes):
            if i == 0:
#                 in_features = prv_layer.in_features if hasattr(prv_layer,'in_features') else 1280 in_features = 
#                 prv_layer.in_features if hasattr(prv_layer,'in_features') else 960 in_features = 
#                 prv_layer.in_features if hasattr(prv_layer,'in_features') else 768 in_features = 
#                 prv_layer.in_features if hasattr(prv_layer,'in_features') else 1536 in_features = 
#                 prv_layer.in_features if hasattr(prv_layer,'in_features') else 1024
                in_features = self.base_model(torch.rand(*(self.data_dim))).data.squeeze().shape[0] ## RH EDIT
            else:
                in_features = pre_head_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=pre_head_fc)
            self.add_module(f'PreHead_{i}', fc_layer)
            self.pre_head_fc_lst.append(fc_layer)
#             if i < len(pre_head_fc_sizes) - 1: non_linearity = torch.nn.ReLU() non_linearity = torch.nn.GELU()
            non_linearity = torch.nn.__dict__[self.nonlinearity](**self.kwargs_nonlinearity)
            self.add_module(f'PreHead_{i}_NonLinearity', non_linearity)
            self.pre_head_fc_lst.append(non_linearity)
    def init_posthead(self, prv_size, post_head_fc_sizes):
        for i, post_head_fc in enumerate(post_head_fc_sizes):
            if i == 0:
                in_features = prv_size
            else:
                in_features = post_head_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=post_head_fc)
            self.add_module(f'PostHead_{i}', fc_layer)
            self.post_head_fc_lst.append(fc_layer)
#                 non_linearity = torch.nn.ReLU() non_linearity = torch.nn.GELU()
            non_linearity = torch.nn.__dict__[self.nonlinearity](**self.kwargs_nonlinearity)
            self.add_module(f'PostHead_{i}_NonLinearity', non_linearity)
            self.pre_head_fc_lst.append(non_linearity)
    
    def init_classifier(self, prv_size, classifier_fc_sizes):
            for i, classifier_fc in enumerate(classifier_fc_sizes):
                if i == 0:
                    in_features = prv_size
                else:
                    in_features = classifier_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=classifier_fc)
            self.add_module(f'Classifier_{i}', fc_layer)
            self.classifier_fc_lst.append(fc_layer)
    def reinit_classifier(self):
        for i_layer, layer in enumerate(self.classifier_fc_lst):
            layer.reset_parameters()
    
    # def forward(self, X, fwd_version=None):
    #     if fwd_version is None:
    #         fwd_version = self.fwd_version
            
    #     if fwd_version == 'head':
    #         return self.get_head(X)
    #     elif fwd_version == 'latent':
    #         return self.forward_latent(X)
    #     elif fwd_version == 'base':
    #         return self.base_model(X)
    #     else:
    #         raise ValueError(f'fwd_version {fwd_version} provided is undefined')
        
    def forward_classifier(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.classify(interim)
        return interim
    def forward_latent(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.get_latent(interim)
        return interim
    def forward_head(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        return interim

    def get_head(self, base_out):
        # print('base_out', base_out.shape)
        head = base_out
        for pre_head_layer in self.pre_head_fc_lst:
          # print('pre_head_layer', pre_head_layer.in_features)
          head = pre_head_layer(head)
          # print('head', head.shape)
        return head
    def get_latent(self, head):
        latent = head
        for post_head_layer in self.post_head_fc_lst:
            latent = post_head_layer(latent)
        return latent
    def classify(self, head):
        logit = head
        for classifier_layer in self.classifier_fc_lst:
            logit = classifier_layer(logit)
        return logit
    def set_pre_head_grad(self, requires_grad=True):
        for layer in self.pre_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad
                
    def set_post_head_grad(self, requires_grad=True):
        for layer in self.post_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad
    def set_classifier_grad(self, requires_grad=True):
        for layer in self.classifier_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad
    def prep_contrast(self):
        self.set_pre_head_grad(requires_grad=True)
        self.set_post_head_grad(requires_grad=True)
        self.set_classifier_grad(requires_grad=False)
    def prep_classifier(self):
        self.set_pre_head_grad(requires_grad=False)
        self.set_post_head_grad(requires_grad=False)
        self.set_classifier_grad(requires_grad=True)



def make_model(
    torchvision_model,
    n_block_toInclude, 
    pre_head_fc_sizes, 
    post_head_fc_sizes,
    head_nonlinearity,
    image_shape=[3,224,224],
    fwd_version='head',
    **kwargs,
):

    ### Import pretrained model


    # base_model_frozen = torchvision.models.resnet101(pretrained=True)
    # base_model_frozen = torchvision.models.resnet18(pretrained=True)
    # base_model_frozen = torchvision.models.wide_resnet50_2(pretrained=True)
    # base_model_frozen = torchvision.models.resnet50(pretrained=True)

    # base_model_frozen = torchvision.models.efficientnet_b0(pretrained=True)

    # base_model_frozen = torchvision.models.convnext_tiny(pretrained=True)
    # base_model_frozen = torchvision.models.convnext_small(pretrained=True)
    # base_model_frozen = torchvision.models.convnext_base(pretrained=True)
    # base_model_frozen = torchvision.models.convnext_large(pretrained=True)


    # base_model_frozen = torchvision.models.mobilenet_v3_large(pretrained=True)

    base_model_frozen = torchvision.models.__dict__[torchvision_model](pretrained=True)

    ### Make combined model

    ## Tacking on the latent layers needs to be done in a few steps.

    ## 0. Chop the base model
    ## 1. Tack on a pooling layer to reduce the size of the convlutional parameters
    ## 2. Determine the size of the output (internally done in ModelTackOn)
    ## 3. Tack on a linear layer of the correct size  (internally done in ModelTackOn)

    model_chopped = torch.nn.Sequential(list(base_model_frozen.children())[0][:n_block_toInclude])  ## 0.
    model_chopped_pooled = torch.nn.Sequential(model_chopped, torch.nn.AdaptiveAvgPool2d(output_size=1), torch.nn.Flatten())  ## 1.

    # image_out_size = list(dataset_train[0][0][0].shape)
    data_dim = tuple([1] + image_shape)

    ## 2. , 3.
    model = ModelTackOn(
    #     model_chopped.to('cpu'),
        model_chopped_pooled.to('cpu'),
        base_model_frozen.to('cpu'),
        data_dim=data_dim,
        pre_head_fc_sizes=pre_head_fc_sizes, 
        post_head_fc_sizes=post_head_fc_sizes, 
        classifier_fc_sizes=None,
        nonlinearity=head_nonlinearity,
        kwargs_nonlinearity={},
    )
    model.eval();

    model.prep_contrast()

    if fwd_version == 'head':
        model.forward = model.forward_head
    elif fwd_version == 'latent':
        model.forward = model.forward_latent
    elif fwd_version == 'base':
        model.forward = model.base_model
    else:
        raise ValueError(f'fwd_version {fwd_version} provided is undefined')

    
    return model
