import React from 'react';

class Waiting extends React.Component {
    constructor() {
        super();
    }

    render() {
        return (
            <div className="spinner">
              <div className="bounce1"></div>
              <div className="bounce2"></div>
              <div className="bounce3"></div>
            </div>
        );
    }
}

export default Waiting;
