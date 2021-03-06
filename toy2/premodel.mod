��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq cmodel
myPTRNNModel
qX?   e:\unnamed\fudan\prml_qiuxipeng\hw\pj\prml2020_pj\toy2\model.pyqX�  class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2048, 2)
        self.sig = nn.Sigmoid()
        #self.soft = nn.Softmax ( dim = 1 )

    def forward(self, x):
        '''
        Please finish your code here.
        '''
        #print ( x.shape )
        x = self.dense ( x )
        x = self.sig ( x )
        #x = self.soft ( x )
        return x
qtqQ)�q}q(X   trainingq�X   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _state_dict_hooksqh	)RqX   _load_state_dict_pre_hooksqh	)RqX   _modulesqh	)Rq(X   denseq(h ctorch.nn.modules.linear
Linear
qX8   E:\anaconda\lib\site-packages\torch\nn\modules\linear.pyqX�	  class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
qtqQ)�q}q(h�hh	)Rq (X   weightq!ctorch._utils
_rebuild_parameter
q"ctorch._utils
_rebuild_tensor_v2
q#((X   storageq$ctorch
FloatStorage
q%X   1666815208704q&X   cpuq'M Ntq(QK KM �q)M K�q*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   1666815206976q2h'KNtq3QK K�q4K�q5�h	)Rq6tq7Rq8�h	)Rq9�q:Rq;uhh	)Rq<hh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBX   in_featuresqCM X   out_featuresqDKubX   sigqE(h ctorch.nn.modules.activation
Sigmoid
qFX<   E:\anaconda\lib\site-packages\torch\nn\modules\activation.pyqGX  class Sigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return torch.sigmoid(input)
qHtqIQ)�qJ}qK(h�hh	)RqLhh	)RqMhh	)RqNhh	)RqOhh	)RqPhh	)RqQhh	)RqRhh	)RqSubuub.�]q (X   1666815206976qX   1666815208704qe.       h�?�;�       f��=*�=K:>cm�>��h�Poe�] �^B>7f�ݗ��\.��>!�?��>���5��������>���=�`�K�>�m��N�>�J�W��>Ρ羆���bm>4F�>�m����>g��?�1>��.w�>,M��p�	>�>�u�>�q� �?\��~�W��>`g>K��k��>gR=�=>U��'?���Z�>BT����=��2�B>T+�����>���>z)�3�>�#?��~>�ٲ>�ž��
��ٯ�"�>��>�S����>���>��>��>4&*�[�?�?�=?羼�m=��>B�ݽ1�3�`�o>e޾�U����=Mr>�u�?����Q�>�I�>���f�L����=�Q�=2,�q־�R?s�d����B��b���
�����>a�ɾ���5>��?Y�>@���\��>4R��.F��J-���G���IV>��ʾ���>�/2��f ?�� ���(>���>�?ȍ޾�_оm˟>i}�L��>��*�>o�5ԇ��ƌ>rV$?��Z�ߞ/��#�>�PK?�O�>猈�٬?�
�>����<��JC�,�>�]�>U��>"��j��>�[?JY�>�?F(�=������>�M2�g��>���3澙�~��k�>j$���~�<<V�[��=]s�>*I���>��������$0��w��}.?l��>�}�>���>�,��7?wĈ��?W��>mc>[��.��> �G>6�z>�}�=I�>$,ӽ��ž7��>�ϲ>O�n����>E�>u*��ڥ>Ξ;��*]�>x���2��:�N�R@t>Ze�	��>��>P���j��><Ł��X�������>��|��>y���?�%I>�ϝ�}~�>ܯ���ؾk�ȾƉ_>w��>]X�`u���?M��@�>�0;>0?h>�9Ҿa��>���>9!��w��=|�>ҰB>`Y�>ߥ߾�H>�Oۺ�׾r)?^�>�n�>7��>�h5��]�e
���_��^J��:#����>�E����>�_?E�>��3�>k�t�ۈ<�Z	��U����#⾂�>�վ������<���J��Ь�"�?2D�O>h�Ӿ�����%��ro�>�(�4ˆ��ƌ>�n�>�����q>$��_�y>5�ɾ�F���k�^��> ɾ���>��F���?�̾����@h2?�y��?�~�>%���=�>���վ�@ܾ�>�O�����U>�}����ɾs�����>���C���N}̾rG��ٻ�]�޾�ܩ���B>0��>�һ��m��;K ����A>'t�>[������E}�W�2�^��D��R����޾���>�[��+?Z�.?�K�>һi��E�;�R>���+�֏�;1)��ܽ.?L��=��>�l��d��>��8>U�>�f��t"����>�Q��>@���ɒ��v�?]v>�$�>����� �>�B�>���_��=j*�>{7R��s�����>̳Ҿj�T>r3?@�u���>�Z>ۍ�$�꾜��>���>9��=�l!?�~¾ą>�\�(�)��e1��M�>�u;�����,�P��e�>�T�>�C�>���>�u	�pZ����=���=/���.�>ND�=ѭ��Y?����<?k�	?����j#>����J'����>��=�tH?���>��=W�? �>$3N�����g�5],���#�g�����>c��=�9Q�������i���*������ŀ��4�?}�ʾ�H7��F>�����=C�>�>Q�?��?�ǔ��'���X@>5��I0�>����>6��� >wϾ����,�=�y�u��\�~��=�,!?�k?�BG>�0Ծ�,%>���>h���R�h���?#tG>f�V��;��|����?�3W0�Z_���Ž���<I��>o.�>9���=�1�j4?�$i�>9z��V�?JDp>���񄮾�k�@[�>�Ä>};]�m޳�����륾�!�>�������$�.߾V����>�l�=;�?1�>|����Ͼ^n���*�"�!?y'�>f��>dł� �3̾q�>�zS>���6\���<���?�T�> �?[�"�@"4�_ؾ��>$uY>��2�>��>+P�{��>���}y������}-?v54���?9#�E�>�/�%��t��=Tq�=��>z�^?�D����=]�?Z�>���>�Ǫ>��>�?��>/���#�:>!�h�=m��D=��`���:T�R�e�`>�����c	>��<���>,����>z��>�7���I�>�fվ����I�>�� ���0�Q��ca��������>z��� ƾ\4�><�H>YX?��>�q>	վ���u׾e�>Wq��#X=�RV>��=i������3����%����>��>zBT>Dg�>���x�>	�`�1>h��"w�<��c�>�$=?�=}��!?����>�	?��>��P��T����>^�>��>�d�=��=�A?��x=��?N}��V%�3E�>�r�U�=��>wx&=�۞>pK�=����v�>F�>P��>(�`>͵?Sr���پ���=3�c>a�>ǎ��g��<a����<��5�s��>ZO�>Z�6��>�X>�el��y>�����>�ը��K�>۾��>s���Á�n��>�����:�A�>L�Ǿ	�
?��?�7f��:��,W�=+�T>�>�AO=b;���߾�|=�>����]�	?ZB�o�z>���>��=��Q���վ�� ?�|վ����ʏ��#8��������þ�
�>\
ɾx��>l�.�JD?��W=Y�ž����V[��!���u?L��U��W�?:����>�Z�>�r�=A�=˻K>Α>L[>�=?��?wy�{�h>�9��jN۾��R>�?@^W>�>��>A%?�辽�>[ǘ>�m?(��>�?���s� #R>|�?t�>��ݾ�d	?�>}龶1��t۾�T<�~����>`$�>Cׄ=�l��n��<�����?%�>�����-�=���>b5i��Ԟ>뚉>���=����ݙ�>��==�~�>���>E�ɾ�+?�K�"_�>^勾���=��׾�v?R�þ���>w'����>˾b>|G*?����+t�U�p>X"a�̻>�վ�?�<?�y
?+����)?���>hX>\�%>Ge�>�?f�e�,��>6�X�6O���F�>���>���=���Yy>����⼟>D��-{��{Q)?YQҾق�>13�>>��>����&!�?�>�"�^s�>�����>���>��>.�>������>%#�� ?��>�r-?�j�Yy��vh?�� �:�>�m��A>��O?�rt�ė����>i�2�j�p����>�1X���>t�?Fg�w��^�%��'ؾ����Ծ1�)>m��Q�(��> ��>�W?S��8�ĥ�>a�9>���>��>J�۾�DȾ&�a>�<�>_V(?��`>C4���b5?�JѼh�����>�ܬ>Ÿӻ{����-?�>>i��< }>F�?S��>f�4�.�|>��?i�>��>@�>��u>��=?��g>S�?" m=�,������5�>��"?�t>�ъ��͌���>���>n>�g�_>`��>�%�>�Q����������7�>+O�>�?5����׾���
?G�2�\�־�>��>�=ؾ|/���&�>��=$��>%Ⱥ>|g�ه�Lw�,螾����v꛽0}=㋥>����A�>������8��=
ѕ=��>������>���>.��>�D��KOx>DA�>�����>ś�2��>Ϙy��(?����|F��ǫ=*��>��۽�򔾢i���m?ex��qw(?���D��"��>8��<w/߾�Ix>����	o�<�<�Ц�>�]?�p=F�>��Ǿ8β���R������	ݾг�>�>���b(��J�>��x>k��>������u	��Ȕ�TO���^�;�^�>9�̾5F�>Cx>��;>�#Y�뾍<=��>9��A�=��P�E��>s]�}9b�^�ϾL�>�e?��ھ����R����>+m?�H�>����,����>9��>��XQ%����o[�>�o�>���>S'}>��?|?��>��>��f�=�.	>�oS��?G�S> P>�g�>��K�����!�>ǅ�>��>��>�k�S����������>��3;5*��R��F��>d=�@oC�"i���C?\k	?�Lݾ칆�� ?��>���Ʃ���I�>�c�>ymG>H螾����*?E޾y��|�ǾO�>f��>������Y���>���'�������v>�Q�p������>� =lr��}��=���=�ݧ�X��>�,�����%�����>!K�L��>xMþD)C>��>��.�Ľ!�9>�?g�j�1^��-�>Ě>���>?�E>[T
�Rí>�B�zգ>Q���W=-<���r>.������B9����T�>
����e�S�۾���ض��P&�>\�?�>Z���h=�Z>��,?H��oމ�����w?�o�>OHԾ>O�>T�?��o>eH��e�>s�#�W��@����>aK?�T	�=أ>���1۾�䯾�#?xߡ>��G�|�l��4?��޾;��>@y��>�>��9��&��1P����>��?ނ#���>�T�>��v�>9�>���=k0���껾6h?'I�4���"�>��<�B>�A3>�/�>��=0)��%q�>�n����l����f����>���>��2�v��>XS��ڰ���>������>8�?��	�%-������Ʉ�p�>���������Ǽ��������s�A�=,�?:�T>#�оϣ�>�y	?L��>F�=��?Hfҽ�·�U�þ�t�>[\��<�>0�5=��>�� ?����>-���p>����a#����>��$���U����>�u|����>R�5>@����=>�m?�4"�gˀ>�{�>��e�����~�>�YN��澷!���?���>�����վ���f?"���l7�>u+b�J�?���>R#�>e,�#��>X�Ѿ�� ?~��>u'ɾD�.>o�>6o���b���Ғ>D�L�u����3>;ȹ>�P�>9�.��"�>]��>�g��+�>�~�l��>�۾ F<v�������
��?�6?�F>g�>E��>>:��$P��`5m��&оH,�>.�Ѿ�uL�js�>Z?�>iG���þ���'���>���=;����>a�5���O�������U��v�>?g>�7>�I�?�Q�����Ȣ�>A;ؾ�Y?�Ԝ=�\"�(:?/�s=��/�?G?q�>>�;>�>��|����=h�>	?Щf>9\�>ꍼS�>�K�>~&�>{Q�<ZӾ�į�49辴��>@l���=��J-�p�fC��7�>�+��#��}��� �����>Z�UB���>��-����>�}�6%m��g%?%����˨� �u�֢�����>����%r�>��ӥ1=LU�>�	<�i?Qc��\�>�7�>ܩ�>�{y>���T���\ھ�~������>k�ᾼ�ᾶ��>l|�n/Z���>��/��I޾.>�ү�&�	?����Nt�e��>��>&��Q�>�!>��#�*����[���d?���<d+�>��?&`�=�p�>�쓾�H�ֽ������v>� �S�>�?L�ٽan�Z�ӽ(A���|i}����>��@?�R�>R��>pp�>o����>^^>���>��1?�Q����?��>aK=.��>�����$�&$>�Y����������?��Y>ڡ��E�O��n��>!_6>���>C��>���>��B=�j?8˜>�ƾ�>�߾�o_�^���pe����<V{�>��C?�B����۾~ܤ��|>�?�Q�<�G�~f?�u�>V>�w�>!�>7�������q>��Z�T��>�ʮ>���<���O☾��c��;�>Hi�dE�>�?�}�>'*�>oY�=�_!>$b3>F�>澓ɵ��p�ƙK>v��i?����/���
���ؾ^��M�����(��%헽�!6>�:�?�����>d���c�q�H��>���>����>�$>��?��ȾP��f?�Sc�mz����>P��>� Y>֣�������	�>4�y>���><O?�rپ]�Ԭ-��:=t�?�߾h-�����:>�H���������>W��>�^>=f��g芽K��>�}�<"l>�Y?Bb�>4Z�>�V?�3�F"�<���>$�>s�;��e>i0��a G���>v��x��>d�޾����o=?I�>[_>�>�3}<���3?B�վL����>+��>ˢ?{�y>�ƾ����#��O��?���>aʈ�!?�l�Mݭ��:?a��������Ҿ9�ʾ�ǐ���r^�=��R��wھafؾL�*���.!C�[�>ｿ�T6�=׾l��>6 �=Tj�>{�U
�>��k)�>'
?���>��پ�ž����� �A����6���J����uV=���>�k>�.D>�.;<�Â>-3þ�^�=	��Ŝ�c�����X��'�����7�q��)�-Gb>o��>��>�ھ�с�(*�3R߾^<->-�K�6d<�K��>5�*@I=��>���>o�ҾY�����>�þ¯��ٌ>%����ז��ݙ>;��ܾ��j�'��R����>>��>M���� پt��>l>>����Yk ?h��R�>q@�`/�d��[P(�D�=S+�>Da{�\��>�����#?��𾭀�>_�R��m�>�"���zq>��>��tu�>"Qb>N��=�><ʭ>6�?����<=��(?��>�z�>?r޾3��>���>`��>5�?/�ν�T�>�ȽY��>��y=�Z?��$���>��,�����l>�^C>��S��>���i�*�b>���qڃ>�zz����>�}���ľU�޾�Y�=�gɾѥ����>��5B��֯�>q��>�=Z�վ�su���?�h=�>��?�E�$�>z���6r�Ůþ���cý�T�>����2����H�]�d�����>Ƹ�>*���=��>��+s
>�\�%|?T����]�>q�?>+�=o�?��>�9
?�R�<j4ؾ1�� /�<�	�1���W1���>���>{I����f�A�y= ���'��@�&?�����>{�^�$������>������'��6%?��=?7���Ǿ)H�>��~��Z�;��N�>��8>�d��˾���>�*�>"h�>4>�+<�>�=���&��>+?x��ɢo�S|���瀾Y7B�S�>�D澰GF�Sd�~��K���,>��3�Y�>q������IH>�,�oS��w4�>�ș����>���>�N�>E��3��p�	i��ߓ�>gž�ǔ>��u>���hC˾��e��a���>,^?�7A�f��>6��b��>c��>�d�>��H>����S�>h��o����H�����Z��>�)?��7��t����>U̾i�����_>�n�>���>�j>�~��FEѾ�?Jv����E:?���>�%�>����U��Qe�p���0?�a������A���\�Pj�>����,>��g>~R>�<@>�� ��A������n0u����Ṙ:G5�>���=c�ݾ�O��,a=�þ��䂾�L2��"(>o ��^?�>�>ڈ?);�>IN���u��  �>0̩>��ӽ{��>�A���搾��P=���&��Q�>�>;�>��R>��Yܜ>�ʾ�v�>�Ɉ��C������<4J���h>�/M>Ӷ1��೾�J���7�>�8��b?�%���Q�>*�"?��>5ξ���>=j��*ؗ�=�����>���׎>����
>�Qľ�Ⱦ�1?Zo�h��)>�5׾R�ؽ�d��?�����M`�>�n>���>�B�E5}=|k�<7:6>�оdF�-��a��>ѹ,?���>r���e�ֽ� L<?j����=񍿾�7>��ܾN��>#�>�V��[W־z�?�>����>2��~&������$>?}?�Y���Ⱦ�?+�):>/`m>T��۞~�q�>�@྄(�<�-��rR>���>iء����>�'���>� <���>b>�-r��y�>d�ɾc������c�����>)7?��>��-�����A>�i������
�8�����=�9��6�b��u�>��j�vqƾ��=K�/>"����>�j>=�7>)yc����=(l�>��ݾU�����>��3=~qҽ��ҽ�d�>���>��
���>���>R��>�S�>�p�:�<��y��>>m�>��0��,��	о��?3�㾏u�>^��>M�>BW�>waq�b��>I�r�>�����T�=��$��վIj&�di�>�_�>�婾pt>�Xپ��>�f�8 >i�>�����%���7>�1?���H��	�X�=F,!����`q�>��>J->�ξ�W���\����?/�������Z	�奂����>��񾍕�\Ⱦ\�C=��>Z{>�ݾ:�>�����&=���꬛�(g>V9��~I!>��>R�3?�H,=�D4��֡�'B����m^?���k�L=J��O�ݾա�0<?I�����2��)p�Q��⾼��=���>S���@���h�l>����Eu��4��>F设Q�>��?��پ��=D��=JR>�s���X3=<}ʾ�������>��Si>�>d?|V˾l>h���E�>�F���B��=ܾfj�>�>�"�>�R�UNξ�O=��?uW�8D=9���GrK�IfQ�!9�>pU���H�����>�5��lʾa�/�Vþ�8�>�wq�[�<aK�>�����ɾ)����߾#_;>���=z�)>��Y>If0>��8>,�y��%�>��Ͼ�<b�+߽~�>�X����^>�?>���>���>��>�g�3�>#O	?��>�&�>2�L?��>���D?�������>�G�=U;�>�B ���?�>�����価�?�1q��-$>Sqs�cG�>��>i�>�u˾���>�P�V &?��?Nm��|L�>d�=��4��a]>P��┺�2r?����M�>8��>��>*��<$3>H�8�2��<���>�w�>"���q'(>o�>�E�>=l?��>�^�>Md�>��?�A,ʾ~g�>)��= �>� �>��K�uYϾ�&�>ߺ�>j�t>�J3>�_�;�z?���>/��>褌���=�D,��5�q���IS>�G?2R��?�B>4H�<_�>�3����Z��=�XϾ*`B��"򾚹�>�6�=G#��mP?�r����>�C�>9��&	�f�o��g�>A���o��.��>������+B>=�>F������>�S�/`�l�u>kþQ�S����=:��>�9���򾅍��������>[��`��>��*?3?����6�����>?O>+=v׾�uվ}]Ǿ����
�=}�>�xͽ���?����a�ֽ�?�7��-e�>������:�>U��O��>{>�5���7w�ڢD���羈����R������\>���>��g>W�1?a�!?�{�>ٸھp���I�j>���>j��=9r�>��?ۗ�>���>�?#���>!>$"�5�;��̽���`ꐾ�$��%�Ԏ>S��=��5���>����>6D��zH�>[����>�4�>N2��Ǉ�<H��> _�>ǎ���5#�����6���>ת!�� ���N?ʢl>�|�>�C��L=�hCb>l�:?���>��L>�:,?�o�>���=[X~;����l�iU�>D�(>78A?��޾���=�����J?���>Uq>�׾�<���7`>sƺ>դ>;/�>�T־�?�a�>��>�1�>�u�>@�>����oýrR��<��C� ?��>�l�>�?�]�����;#�����<
��>VS�>9����]�5�?�3S>�qD>~��j�NR��?$?\>q�>���V4�;�>�k��2(�.�?U�žܢ>l��>�P�=�>F�3�&B.?w����|ƻ�u�d�0?��>��1��k�$���x
�=��>���>*��D��Fl�n��Nt�����D1�TI�w��>�F�f0m>���=[��> b >��Z>z�=@�5,?Ҧ����мb/�����>㳺�8�ľ+4�>u�����>nC�>�?���� ?0�>m�?�>݊ ?[bݾe�>s	�>�x�\Kc�c+������b��%�>%�?� �>|�¾�4�=/?�e�<�+��l�=�)?�c�>�=J[���ӾwS���Ҿr�?7/��uE?�p���>�����?;Η�rk8�н��L�?�%�m��>��Ҿ>D��Ƚ�WO>��C>%��P����w���%�⽢���^��Nz��D�>ͱ+>Hщ���?�8�-��;p$���������>ց��>�ݾ�ϊ��[�V��DԺ>ɮ�>}��ֿk���޾�c�>��>�-�>ˁ�<�@>����+̾�^�:���>P��CQ>&�x� �?�b;M�=QҾ��>�B��L��>��g>Y���>'9�:5���j&�>I�	�P�Ւl?|�>,>ǽ.S�2�ӾI]�\5�>�>L>6������ł>/��&l.>��s�^5����|�J�T>S��>.��=kS��V�>vX�>���>��>6	?�6�>�	���ǿ>Ȃھ��*>����i�q��>�>�>Bk�=%��>l�10�=q��>�y��ѭ=᝗�U�����d>�ꚽT�H��?��DG�M%�>z���>�`����>�^�>��Y��	���[����ҟپvX�[��>��ľ)��Mg��<\��q �=��>�rX�K�!�AXǾ\0�>����Q��JX�>-U�>-D�>�����|>&�����4�8���?�;?Jb�>a/ �����>%���兾�Gk>w���D������7��>����L��c羓B}��a�>�%���=�J��z�>�)r�zm�>.%�v�>S���b!?P���T�>�?o�g�.�jJ�>���=��d�"=b½_��>^�	����4���$=0 >��5�z~���|P��4/��.׾ׁ���]>f�?[	'�j)P>Ĵg>V�ЋԾ�L��pI?�G2�ŕ�=`a�����>�N�>�4%�jI�>/#��`羮�ʾ���>Ή
>-�վ�r?�����k>�����о:���H_����>A�A�?\����+о͞$��y=��>��Å�>hu׾Go�>� �>�U�+%�>#��>܃ʾ]�1?�Aj>�뾏@>�f��h ��>b0�>��!?H�>U/=u�>��?�`=�>8>�����Ծ������>>aD��=�D��Ľ��¾�)�>t$�>�mZ�{u��{2!���D��V�>_�/����<&��>�j��d���ێ=��>H�7���E����>v↾89
��V��rE.>�:v��������������s�u�.8��Vk��r�K���s�>ߝ?=x���'��D�l�me�>�v�>L�ɾ�W��%\��{>c�h�c���c3����>%�{>y��>8{��~�ʾN��m�>���>�!�>�4�Z*>4H�>����=���P�>b?8������,`��Og��K
>�?#\�>/�>y��>�ه=vqX��������>��jX�>�X�<}$�V�����ܾ��?�V��2�ҾB��6�>����B��t3?4̙����>��Ҿf]~>^  ���>1�d>/b��
�پט�=e�>#��>-o���j>��*��A�>���>b�ƾT��<<�>�	���p�>��|>u�?>��ܾ�� �V&�?�����>���>{XS>��>�>
�¾$��=ي$?0����Fn�Oq�� ?�'�>@�?���>�T�>\���Ⱦ6��>�}����\��]��2U>H�X�����K?�S��vaP>����X?�c>Խ�>u}��OI�O��>HT�>.>����"�	����={�=*�;> �{h��A�
?Eu+?г�>���/޾�'⾓h�p��������ؾ���>������պM?;S��\�qi
�ym�� ?!��=��>G����龡��@�ᾴH	?���>���>dm��D�<�A?��>
��ԾG>uM]>���>?��
����>��>Ѣ���=žq�>@/�=�2��B����W�y�>��>�y�%T�>�?;��>����/���Ό?_�o>��B?ra#?�dE<iAq���Q>��=;���'�ب�>��;ʽ9Ø>D����)?�?��&?!H�>Y�/TO>B�KB�>$=��Ǿ�}�>�V�=2])�P���sd>�Ā>Q~"�Y0���r���J0��
?�⧾$�">ET��M�>+?��{nw>��u�(��:lC�>2&!>i�(=~ʾ���>"��>v��>9��>���>�ʸ�oW��󭾃?�h��\72��/�W�>��>��>�y ���߾2��>�ê�R�̲���@�>Zb���R ?��>3y?�	��x��R?N7����!>:��>�Q�>>	!��ة�~me>)�f>]P��o�>͍�xM�>���&O>BW�>�?����� Y?�D��?u���u�<`P��'g���>5F�>}����=9�>�d�b�>��X�H�,=�	\����c��?�Ǿ��>�K[> ^�>��?��)�B���X13?�������>~�>����"�>vڵ�ױ�R�;&b�>�f? x�=�ŷ�"��<(�"?�<��?yH�>��>�����ppU����>0پ^���̾�����u��=0�>�>hD���>����������W���?�!6?q`���>�?��۾Q�$>8f>�ݾ�Ox>����=��~�>�M��9�Z >����Ͼ��='��>>�0>��>�/�>+�>*���C��-@�>W��>T�>�H����>�竾v�Y>�m��䙾�3 �a�?��ž���>�\��s槾�ľ>"5�k���)��>��>�[���>>"�><��P��u��� �/?8�Ͼ����?8_��@o�>2���:�>6Y�<d:�>{5?��?����!�6o7�IM��վ�ˁ>�ۚ>_L>���>����f�>{�E>!���	d?u��>M�>��?OM��7��{��>3�����>A?lڿ�.�>��U>���V~K�Oc!>}L�T��>W��=�ӡ�;��>��f�����8>�����O��^%>��؈� ���"�j�>��=�+��������I+S��9Ⱦ���<����� >uu��þ�u<��>h�>S[�>H�ƾ�,�>F>��/?��?��<>����'?��>5�?J��<�,�=B4?��>�ܽ�e0?����@�>=G�=J�(�`��>���>��\>zG�=�-���r�>���vo�>�]��14�����:3>����>v�¾�Et���������?�H?���>~��>�o?;ظ��|�>��>Z�ݾ�G	?H�Z>�����,=�>W��4��>����
q=�Mr>�ܾ?7��*��<����P�Q�>©-?�a�>~
?~	���>/U�;�$���gux���ƾvB�>~��6z�=z�?��h��.?�;-c��=�?���={�>1�f>CȂ>��¾��?�����!���� ��~��>���T]��N�����=�|����>�k����Y�D� [�����>��0?3�&��Ҽ�x?�;�>]A��!M��]�>n�?�?����@8������C޾�f���A�����S��>k}��o��>E�u>i�>!��>�n;e�wD����=�R�>��>����t�F�?_�0>)_��M����?��о��ﾱ�>w[>�`}�f�=�t�5:��S�9?t�>��d>je��@�?EF־1���,�������\����B�L���>�J�>p�>�M���?���?��2?�@?1�>�:�>h�:>�:�>=l!<i��=�4���>>���=�L��;�x>�f>���'վ�7�>e����`1��L�O�>��>Y��QAW>��>%Z���վ~D�x6�>fޘ>	��>辰�����O|���j�5G�>��>�]6?�x��g����>���=���>�,���u>���=���>����}�x�{D�>p�=��Ⱦ)TȺ8�`�ǒ�n>���ɿ��L�!n'>aj&����w/�j�=?�T%�w.�>�pE>����n;�>�t龃V�>�%�>��8��ڳ����e�����=z-?>e�3�E��>�}�>�	�	;þ/��3z��>��>D��>H!�>�h�������>�^��E�>��>���p�?��> ��>I��>|��>h�>3I����H>Ie�>j��>�9>�(>�@?kk�4��>�*S><��>��ླ�Q6׾���>�ջ��;�>-�о�t�2�ʾ���>�Y�>�c>�4 ?]�>L��>I��>;p�=Z�!�J}ھ��6�(=S��AU�|䆾�N�>��Ͻ���>���={R�>U�?	�?��>W�>��o>���=� m����3a���>�E�;��>ū�>�i%��gF>|0>ݻ��"=�5#�y��3�Ⱦ���>*�>�g� ��>Ｗ>z�c6�>;�>aכ����>z×=ޭ�<�4�>	�#��Ob�=��>yt��~+u=�V�>����#>�❾/�=>�i�>��>X�%?���)n�A�n>Y�Ћ�=��%��r�>Fپ�`>bsľ���>[}n��ݾ��?A���[i�6L�,���<������HQ�>��~���'����
yԾ���>��þ+n׾�8־���%��=�ܰ�=l��x�5���%�$>�;�޻]>��
?[%����5�FD�>UB�����>Wl�:�g`��h?�i��N�=F�Ҿ�T�>0�>�w�>�#����>�J?���Yr>��>]H��4��jK��t�>�?�>�\���(���R=���>b�ݾRi?��m>��>`�=Օ>e���a�=mu?6�I>J`�><ò>�a����ľ���>cj����?���i_?����0T�>w㽾K@�a-���9�U�ž��LÕ;ã�>U?��2?C?��>s��>�����ĵ�m�>k�G>�@�}!�>�η>?�,� NP=-O����S>:�>������><Q(?�"�q�ڽ1?6o�>�����<@�>�=�;�����=��=���>�і�Y��侄1��W<���/��>J}��;+��Z�>jp>�+�>�u>�HS>Ciy�>9�>`I>ݫ
?�D�>��>���k�?������?_m�=h�1�.q>�؇>�EоÆ=bh�i^��O��~��>Z?AZ?'�=�\�ۓ�>�������^+�>If�>Ifk>�?×��:����/>LS��C�?M�¾��ž��׾7�V��*�=yľ*®>��>���>���>��n�\'��B=�<	>�9���>˛�>��i��{��� ܾx�^��m�=��>���NZ)?O�=����T2�u@���+�>�A|>�h>t��>;7+�H�>�?uR�=�B>�OԾN�?�#�I
g�/�B�W��C�?���=ߌ�>��>g_?{�s�ğ������>�>Դ��`t�>|N�>L�}>}f@>����p�>���Z5�w.�N<�����>jA<�禾���J9�=��ȾU�>ތ�>��S��?��>�>��İ�=&"ǾUM�j2�>�2��`��>�ݾ���>���>U�>~�@>�f���j�5	=��>�q�>e��X�8?������=c�޾ �#������T�>��0��>K��=d�?k1��<2�>]�����?�����>���>�����>{>�k�>��Y��>