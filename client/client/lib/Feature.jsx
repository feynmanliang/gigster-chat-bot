import React, { PropTypes } from 'react';
import Message from './Message.jsx';

class Feature extends React.Component {
    constructor() {
        super();
    }

    componentDidMount() {
        const { feature } = this.props;
        feature.updateElt(this.refs.feat);
        feature.createGraph();
    }

    render() {
        const { index } = this.props;
        return <li ref="feat" className="feature"/>;
    }
}

export default Feature;
