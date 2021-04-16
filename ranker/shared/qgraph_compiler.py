"""Tools for compiling QGraph into Cypher query."""


def cypher_prop_string(value):
    """Convert property value to cypher string representation."""
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        return f"'{value}'"
    else:
        raise ValueError(f'Unsupported property type: {type(value).__name__}.')


class NodeReference():
    """Node reference object."""

    def __init__(self, node, anonymous=False):
        """Create a node reference."""
        node = dict(node)
        node_id = node.pop("id")
        name = f'{node_id}' if not anonymous else ''
        labels = node.pop('type', 'named_thing')
        if not isinstance(labels, list):
            labels = [labels]
        props = {}

        curie = node.pop("curie", None)
        if curie is not None:
            if isinstance(curie, str):
                props['id'] = curie
                filters = ''
            elif isinstance(curie, list):
                filters = []
                for ci in curie:
                    # generate curie-matching condition
                    filters.append(f"{name}.id = '{ci}'")
                # union curie-matching filters together
                filters = ' OR '.join(filters)
            else:
                raise TypeError("Curie should be a string or list of strings.")
        else:
            filters = ''

        node.pop('name', None)
        node.pop('set', False)
        props.update(node)

        self.name = name
        self.labels = labels
        self.prop_string = ' {' + ', '.join([f"`{key}`: {cypher_prop_string(props[key])}" for key in props]) + '}'
        self._filters = filters
        if curie:
            self._extras = f' USING INDEX {name}:{labels[0]}(id)'
        else:
            self._extras = ''
        self._num = 0

    def __str__(self):
        """Return the cypher node reference."""
        self._num += 1
        if self._num == 1:
            return f'{self.name}' + \
                   ''.join(f':`{label}`' for label in self.labels) + \
                   f'{self.prop_string}'
        return self.name

    @property
    def filters(self):
        """Return filters for the cypher node reference.

        To be used in a WHERE clause following the MATCH clause.
        """
        if self._num == 1:
            return self._filters
        else:
            return ''

    @property
    def extras(self):
        """Return extras for the cypher node reference.

        To be appended to the MATCH clause.
        """
        if self._num == 1:
            return self._extras
        else:
            return ''


class EdgeReference():
    """Edge reference object."""

    def __init__(self, edge, anonymous=False):
        """Create an edge reference."""
        name = f'{edge["id"]}' if not anonymous else ''
        label = edge['type'] if 'type' in edge else None

        if 'type' in edge and edge['type'] is not None:
            if isinstance(edge['type'], str):
                label = edge['type']
                filters = ''
            elif isinstance(edge['type'], list):
                filters = []
                for predicate in edge['type']:
                    filters.append(f'type({name}) = "{predicate}"')
                filters = ' OR '.join(filters)
                label = None
        else:
            label = None
            filters = ''

        self.name = name
        self.label = label
        self._num = 0
        self._filters = filters
        has_type = 'type' in edge and edge['type']
        self.directed = edge.get('directed', has_type)

    def __str__(self):
        """Return the cypher edge reference."""
        self._num += 1
        if self._num == 1:
            innards = f'{self.name}{":" + self.label if self.label else ""}'
        else:
            innards = self.name
        if self.directed:
            return f'-[{innards}]->'
        else:
            return f'-[{innards}]-'

    @property
    def filters(self):
        """Return filters for the cypher node reference.

        To be used in a WHERE clause following the MATCH clause.
        """
        if self._num == 1:
            return self._filters
        else:
            return ''
