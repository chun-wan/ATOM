filepath = '/home/ginsongsong/workspace/ATOM/atom/models/gemma4.py'
with open(filepath, 'r') as f:
    content = f.read()

old = """    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.hidden_size**0.5)"""

new = """    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.hidden_size**0.5)"""

if old in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print('PATCHED_OK')
else:
    print('OLD_NOT_FOUND')
    idx = content.find('class Gemma4Model')
    if idx >= 0:
        fwd = content.find('def forward', idx)
        if fwd >= 0:
            print('FOUND_FWD:', content[fwd:fwd+200])
